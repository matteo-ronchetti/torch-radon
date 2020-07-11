#!/usr/bin/env python3
# pylint: disable=fixme
r"""
This module (``AlphaTransform.py``) provides the class
:class:`AlphaShearletTransform` which can be used to compute the alpha-shearlet
transform of images.

The parameter alpha determines the *directionality* of the system:

    * alpha = 1 yields a wavelet-like system,
    * alpha = 0.5 yields a shearlet system,
    * alpha = 0 yields a ridgelet-like system.

The following is a simple example indicating how the transform can be used.
For more details, we refer to the documentation of the class
:class:`AlphaShearletTransform` below.

::

    import numpy as np
    from AlphaTransform import AlphaShearletTransform as AST

    # create a transform for images of resolution 600 x 500,
    # with alpha = 0.5 and 3 scales
    my_trafo = AST(600, 500, [0.5]*3)

    test_im = np.random.random((500, 600))

    # compute the alpha-shearlet coefficients
    coeff = my_trafo.transform(test_im)
    # threshold the coefficients
    thresh_coeff = coeff * (np.abs(coeff) > 3)
    # reconstruct
    recon = my_trafo.inverse_transform(thresh_coeff)
    print("Reconstruction error", np.linalg.norm(test_im - recon))

Apart from collecting all settings/parameters in one object,
creating `my_trafo` also does any necessary precomputation
(e.g. of the alpha-shearlet filters).
Thus, creation of the instance `my_trafo` might be a little slow,
but computation of each individual transform is comparatively fast.
"""

# Note to self: Subsampling will destroy smoothness w.r.t. alpha, since the
# subsampling rate depends on alpha.
# Possible solution: Only subsample in one direction
# (x direction for shearlets on the 'usual' cone).

# Note to self: One could possibly remove some of the indices belonging to
#               transitions between the cones, since these occur (essentially)
#               doubled.

import math
import numpy as np
import numexpr as ne
import pyfftw
# import numpy.fft as fft
# from numpy.fft import fft2, ifft2, fftshift, ifftshift
# from numpy.fft import ifftshift

# import MotherShearletsTheano # this must be before 'import MotherShearlets'
import MotherShearlets as MS
from fourier_util import fft2, ifft2, my_fft_shift, my_ifft_shift
from tqdm import tqdm


class AlphaTransformException(Exception):
    pass


def div0(a, b):
    r"""
    This (numpy-vectorized) function computes a/b, ignoring divisions by 0.

    Example:
        div0( [-1, 0, 1], 0 ) -> [0, 0, 0]
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0  # -inf inf NaN
    return c


def _find_spect_file(spectrograms_directory,
                     width,
                     height,
                     alphas,
                     real=False,
                     parseval=False,
                     periodization=True,
                     **_):
    spectrograms_file = None
    with open(spectrograms_directory) as spect_dir:
        for dir_entry in spect_dir:
            options, path = dir_entry.split(':')
            path = path.strip()
            options = options.split(';')
            cur_width = int(options[0])
            cur_height = int(options[1])
            if (cur_width, cur_height) != (width, height):
                continue
            options[2] = options[2].strip()
            assert options[2][0] == '[' and options[2][-1] == ']'
            cur_alpha_strings = options[2].strip('[]').split(',')
            cur_alpha_strings = [s.strip() for s in cur_alpha_strings]
            alpha_strings = ["{:.2f}".format(alpha)
                             for alpha in alphas]
            if alpha_strings != cur_alpha_strings:
                continue
            option_strs = [str(x)
                           for x in [real, parseval, periodization]]
            cur_option_strs = [option.split('=')[1].strip()
                               for option in options[3:]]
            if option_strs == cur_option_strs:
                spectrograms_file = path
    if spectrograms_file is None:
        raise AlphaTransformException("Could not load shearlets from "
                                      "database: No suitable "
                                      "spectrogram file was found in "
                                      "the database!")
    return spectrograms_file


def _build_alphas_str(alphas):
    return '[' + ", ".join(["{:.2f}".format(alpha) for alpha in alphas]) + ']'


class AlphaShearletTransform:
    """
    This constructor initializes the :class:`AlphaShearletTransform` object.

    Only three arguments are *required*. The remaining optional arguments
    have sensible default values which can be adjusted to achieve specific
    behaviours which are described below.

    **Required parameters**

    :param int width:
        The width (in pixels) of the image(s) to be analyzed

    :param int height:
        The height (in pixels) of the image(s) to be analyzed

    :param list alphas:
        The length of this list determines the number of scales to be used.
        Each element ``alphas[i]`` determines the value of alpha to be used
        on scale ``i``. Hence, it should satisfy ``0 <= alphas[i] <= 1``.

        .. note:: The most common choice is ``alphas = [alpha] * N``, which
                  will yield an alpha-shearlet transform with ``N`` scales
                  (plus the low-pass part), i.e., the same value of alpha is
                  used on all scales.

    **Keyword parameters**

    :param bool real:
        Flag indicating whether the transform should use *real-valued*
        alpha-shearlets. In particular, this means that the transform of a
        real-valued signal is real.

        Setting ``real = True`` amounts to a symmetrization of the
        alpha-shearlets on the Fourier-side.

        .. note:: Setting ``real = True`` is incompatible with
                  ``subsampled = True``!

    :param bool subsampled:
        If set to true, the *subsampled* transform will be used. This has the
        following consequences:

        * a lower redundancy,
        * a lower memory consumption,
        * the transform is *not* translation invariant,
        * the ``transform()`` method will return a list of 2-dimensional numpy
          arrays with **varying dimensions**, instead of a single 3-dimensional
          numpy array.

        .. note:: Setting ``subsampled = True`` is incompatible with setting
                  ``real = True`` and with setting ``generator = True``!

    :param bool parseval:
        Flag indicating whether the alpha-shearlets should be normalized
        (on the Fourier side) to get a Parseval frame.

    :param bool periodization:
        Flag indicating whether the shearlets on the highest scale (which
        (potentially) exceed the Fourier domain), are to be periodized (if
        ``periodization = True``) or truncated (if ``periodization = False``).

        .. note:: The subsampled transform *must* be periodized.

    :param bool generator:
        Flag indicating whether the spectrograms should be precomputed (if
        ``generator = False``) or computed each time the (inverse)
        transformation is computed (if ``generator = True``).

        Setting ``generator = False`` will improve the runtime (if the
        transform is computed several times) but greatly increase the memory
        footprint, in particular for small alpha, large images and many scales.

        .. warning:: If ``generator`` is set to ``True``, the ``transform()``
                     method will return a generator instead of a 3-dimensional
                     numpy array. This should be taken into account in
                     implementations.

        .. note:: Since the subsampled transform is already memory efficient,
                  setting ``generator = True`` for the subsampled transform
                  makes no sense and is thus not allowed.

    :param bool use_fftw:
        Flag indicating whether the ``pyfftw`` package should be used to
        compute Fourier transforms (which are used internally to compute
        convolutions).

    :param mother_shearlet:
        This object determines the mother shearlet to be used. If the default
        value (``None``) is provided, the
        :obj:`"Haeuser shearlet" <MotherShearlets.HaeuserMotherShearlet>`
        will be used.

        .. note:: For more information on other available mother shearlets, see
                  ``MotherShearlets.py``.

    :type mother_shearlet: :class:`~MotherShearlets.MotherShearlet`

    :param bool verbose:
        Flag indicating whether a progress bar should be displayed while
        precomputing the shearlet system.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self,
                 width,
                 height,
                 alphas,
                 *,
                 # num_shears = None,
                 real=False,
                 subsampled=False,
                 generator=False,
                 parseval=False,
                 periodization=True,
                 use_fftw=True,
                 mother_shearlet=None,
                 verbose=True):
        # spectrograms_directory=None):
        spectrograms_directory = None  # feature currently disabled
        # We begin with several sanity checks (assertions)
        assert all(0 <= alpha <= 1 for alpha in alphas), ("alphas must be "
                                                          "between 0 and 1.")
        if subsampled:
            assert not real, ("In the current implementation, a real-valued "
                              "transform can not be subsampled.")
            assert not generator, ("Since the subsampled transform is memory "
                                   "efficient, setting 'generator = True' "
                                   "makes no sense and is not supported.")
            assert periodization, ("The subsampled transform MUST be "
                                   "periodized, given the current "
                                   "implementation.")
        if spectrograms_directory is not None:
            assert not subsampled, ("Currently, precomputed spectrograms can "
                                    "only be loaded for the fully sampled "
                                    "transform!")
            assert not generator, ("The generator feature can not be combined "
                                   "with loading precomputed spectrograms!")

        # Set the relevant parameters
        self._width = width
        self._height = height
        self._alphas = alphas
        self._real = real
        self._subsampled = subsampled
        self._generator = generator
        self._parseval = parseval
        self._periodization = periodization
        self._use_fftw = use_fftw

        # for caching the values of the L^2 norms of the individual shearlets
        self._cache = False
        self._frame_bounds = None

        # NOTE : we do not count the "low pass scale" as a scale!
        # self.__num_scales = len(alphas)
        # calculate self.__x_min, self.__x_max, self.__y_min and self.__y_max
        self._calculate_bounds()
        self._indices = AlphaShearletTransform.calculate_indices(real, alphas)

        if spectrograms_directory is None:
            if mother_shearlet is None:
                # self._mother_shearlet = MeyerMotherShearlet
                self._mother_shearlet = MS.HaeuserMotherShearlet
            else:
                self._mother_shearlet = mother_shearlet

            if self._subsampled:
                self._wrapped_to_coord = []
                self._wrapped_to_index = []

            # calculate the spectrograms of the filters
            self._rescale_filters()
            if subsampled:
                self._calculate_subsampled_spectrograms(verbose=verbose)
            else:
                self._calculate_full_spectrograms(verbose=verbose)
        else:
            # first, find the right file
            spectrograms_file = _find_spect_file(spectrograms_directory,
                                                 width,
                                                 height,
                                                 alphas,
                                                 real,
                                                 parseval,
                                                 periodization)

            # then, load the actual spectrograms
            spects = np.load(spectrograms_file)
            self._spectrograms = [spect for name, spect in spects.iteritems()]
            assert len(self._spectrograms) == len(self._indices)
            assert np.all([spect.shape == (height, width)
                           for spect in self._spectrograms])

        self._setup_fft()

    def _save(self, spectrograms_directory, spect_path):
        np.savez(spect_path, *self._spectrograms)
        with open(spectrograms_directory, "a") as spect_dir:
            form_str = ("{w}; {h}; {alphas_str}; real={real}; "
                        "parseval={parseval}; periodization={per} : {path}\n")
            alphas_str = _build_alphas_str(self._alphas)
            spect_dir.write(form_str.format(w=self._width,
                                            h=self._height,
                                            alphas_str=alphas_str,
                                            real=self.is_real,
                                            parseval=self.is_parseval,
                                            per=self.periodization,
                                            path=spect_path + ".npz"))

    @classmethod
    def _append_fftws(cls, threads, shape, fftws, ifftws, normalizations):
        inp = pyfftw.empty_aligned(shape, dtype='complex128')
        out = pyfftw.empty_aligned(shape, dtype='complex128')
        fftws.append(pyfftw.FFTW(inp,
                                 out,
                                 axes=(0, 1),
                                 threads=threads))
        inp = pyfftw.empty_aligned(shape, dtype='complex128')
        out = pyfftw.empty_aligned(shape, dtype='complex128')
        ifftws.append(pyfftw.FFTW(inp,
                                  out,
                                  axes=(0, 1),
                                  direction='FFTW_BACKWARD',
                                  threads=threads))
        normalizations.append(math.sqrt(np.prod(shape)))

    def _setup_fft(self):
        if not self._use_fftw:
            return
        threads = ne.detect_number_of_cores()
        height = self._height
        width = self._width
        if not self.is_subsampled:
            inp = pyfftw.empty_aligned((height, width), dtype='complex128')
            out = pyfftw.empty_aligned((height, width), dtype='complex128')
            self.__fftw = pyfftw.FFTW(inp, out, axes=(0, 1), threads=threads)
            inp = pyfftw.empty_aligned((height, width), dtype='complex128')
            out = pyfftw.empty_aligned((height, width), dtype='complex128')
            self.__ifftw = pyfftw.FFTW(inp,
                                       out,
                                       axes=(0, 1),
                                       direction='FFTW_BACKWARD',
                                       threads=threads)
            self.__normalization = math.sqrt(width * height)
        else:
            fftw = []
            ifftw = []
            normalization = []
            for spect in self.spectrograms:
                AlphaShearletTransform._append_fftws(threads,
                                                     spect.shape,
                                                     fftw,
                                                     ifftw,
                                                     normalization)
            # this last entry is/can be used to compute
            # transforms of the _full_ image
            AlphaShearletTransform._append_fftws(threads,
                                                 (height, width),
                                                 fftw,
                                                 ifftw,
                                                 normalization)
            self.__fftw = fftw
            self.__ifftw = ifftw
            self.__normalization = normalization

    def _fft(self, i, inp):
        if not self._use_fftw:
            return fft2(inp)
        else:
            if not self.is_subsampled:
                return self.__fftw(inp) / self.__normalization
            else:
                return self.__fftw[i](inp) / self.__normalization[i]

    def _ifft(self, i, inp):
        if not self._use_fftw:
            return ifft2(inp)
        else:
            if not self.is_subsampled:
                return (self.__ifftw(inp, normalise_idft=False) /
                        self.__normalization)
            else:
                return (self.__ifftw[i](inp, normalise_idft=False) /
                        self.__normalization[i])

    def _compute_wrapping_indices(self):
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements
        assert self._subsampled

        x_min = self.__x_min
        y_min = self.__y_min
        width = self.width
        height = self.height

        hori_low_pass_support = self.__horizontal_low_pass_fct.support
        hori_low_pass_support = [math.ceil(hori_low_pass_support[0]),
                                 math.floor(hori_low_pass_support[1])]
        vert_low_pass_support = self.__vertical_low_pass_fct.support
        vert_low_pass_support = [math.ceil(vert_low_pass_support[0]),
                                 math.floor(vert_low_pass_support[1])]
        wrapped_to_coord = self._grid(*(hori_low_pass_support +
                                        vert_low_pass_support))
        x_index_bounds = [(x - x_min) % width for x in hori_low_pass_support]
        # as usual: we want to implement the usual orientation of the y-axis
        y_index_bounds = [(height - 1) - ((y - y_min) % height)
                          for y in vert_low_pass_support][::-1]
        self._wrapped_to_coord.append(wrapped_to_coord)
        self._wrapped_to_index.append([x_index_bounds, y_index_bounds])

        hori_cones = frozenset({'l', 'r'})
        negative_cones = frozenset({'l', 'b'})
        dir_fct = self._mother_shearlet.direction_function

        for j, k, cone in self._indices[1:]:
            if cone in hori_cones:
                cur_scale_fct = MS.scale(self.__horizontal_scale_fct, 2**j)
            else:
                cur_scale_fct = MS.scale(self.__vertical_scale_fct, 2**j)
            if cone in negative_cones:
                cur_scale_fct = MS.flip(cur_scale_fct)

            cur_dir_fct = MS.scale(MS.translate(dir_fct, k),
                                   2**((self._alphas[j] - 1) * j))
            beta, gamma = cur_dir_fct.support

            # we compute everything as if we are in cones 'l' or 'r'.
            # in case of 't' or 'b', we need to adjust in the end
            a, b = cur_scale_fct.support
            # DEBUG start
            # if (j,k,cone) == (3,2,'r'):
            #     print("(a,b) = ({0},{1})".format(a,b))
            #     print("(beta,gamma) = ({0},{1})".format(beta, gamma))
            # DEBUG end
            if a > 0:
                para_width = math.ceil(b - a)
                para_height = math.ceil((gamma - beta) * b)
                para_slope = gamma
                para_upper_left = (a, gamma * a)
            else:
                para_width = math.ceil(b - a)
                para_height = math.ceil((beta - gamma) * a)
                para_slope = beta
                para_upper_left = (a, beta * a)

            x_range = np.arange(math.floor(para_upper_left[0]),
                                math.floor(para_upper_left[0]) +
                                para_width + 1)
            wrapped_x = x_range % (para_width + 1)

            y_offsets = np.floor(para_slope * (x_range - para_upper_left[0]) +
                                 para_upper_left[1]).astype('int')

            # NOTE: y is traversed in the array-index order, which is different
            #       from the mathematical orientation of the y-axis.
            #       We fix this later depending on the cone
            #       in which we are (horizontal/vertical)
            y_range = np.hstack((np.arange(y_offset - para_height,
                                           y_offset + 1)[np.newaxis].T
                                 for y_offset in y_offsets))
            wrapped_y = y_range % (para_height + 1)

            # depending on the cone we need to "switch" x- and y-coordinates
            if cone in hori_cones:
                actual_para_width = para_width
                actual_para_height = para_height
                actual_x_range = np.broadcast_to(x_range,
                                                 (para_height + 1,
                                                  para_width + 1))
                actual_y_range = y_range
                actual_wrapped_x = wrapped_x
                actual_wrapped_y = wrapped_y
            else:
                actual_para_width = para_height
                actual_para_height = para_width
                actual_x_range = y_range.transpose()
                actual_y_range = np.broadcast_to(x_range[np.newaxis].T,
                                                 (para_width + 1,
                                                  para_height + 1))
                actual_wrapped_x = wrapped_y.T
                actual_wrapped_y = wrapped_x[np.newaxis].T

            wrapped_to_coord = [np.empty((actual_para_height + 1,
                                          actual_para_width + 1),
                                         dtype=np.int)
                                for i in range(2)]
            wrapped_to_index = [np.empty((actual_para_height + 1,
                                          actual_para_width + 1),
                                         dtype=np.int)
                                for i in range(2)]

            y_indices = (height - 1) - ((actual_y_range - y_min) % height)
            x_indices = (actual_x_range - x_min) % width
            actual_wrapped_y = (actual_para_height - 1) - actual_wrapped_y

            wrapped_to_coord[0][actual_wrapped_y,
                                actual_wrapped_x] = actual_x_range
            wrapped_to_coord[1][actual_wrapped_y,
                                actual_wrapped_x] = actual_y_range

            wrapped_to_index[0][actual_wrapped_y, actual_wrapped_x] = x_indices
            wrapped_to_index[1][actual_wrapped_y, actual_wrapped_x] = y_indices

            self._wrapped_to_coord.append(wrapped_to_coord)
            self._wrapped_to_index.append(wrapped_to_index)

    def _calculate_subsampled_spectrograms(self, verbose):
        # pylint: disable=too-many-locals
        self._compute_wrapping_indices()
        hori_cones = frozenset({'l', 'r'})
        wrapped_to_coord = self._wrapped_to_coord

        hori_low_pass_fct = self.__horizontal_low_pass_fct
        vert_low_pass_fct = self.__vertical_low_pass_fct
        direction_function = self._mother_shearlet.direction_function
        hori_scale_fct = self.__horizontal_scale_fct
        vert_scale_fct = self.__vertical_scale_fct

        spectrograms = []

        # first, calculate the low-pass spectrograms
        spectrograms.append(hori_low_pass_fct.call(wrapped_to_coord[0][0]) *
                            vert_low_pass_fct.call(wrapped_to_coord[0][1]))

        # then, calculate the remaining spectrograms
        # for (j, k, cone), coord in zip(self._indices[1:],
        #                                wrapped_to_coord[1:]):
        if not verbose:
            my_tqdm = lambda x, total, desc: x
        else:
            my_tqdm = tqdm
            # print("Precomputing shearlet system...")
        for (j, k, cone), coord in my_tqdm(zip(self._indices[1:],
                                               wrapped_to_coord[1:]),
                                           total=len(self._indices) - 1,
                                           desc='Precomputing shearlets'):
            cur_hor_scale_fct = MS.scale(hori_scale_fct, 2**j)
            cur_vert_scale_fct = MS.scale(vert_scale_fct, 2**j)
            cur_hor_scale_fct_flipped = MS.flip(cur_hor_scale_fct)
            cur_vert_scale_fct_flipped = MS.flip(cur_vert_scale_fct)
            cur_dir_fct = MS.scale(MS.translate(direction_function, k),
                                   2**((self._alphas[j] - 1) * j))

            if cone in hori_cones:
                scale_coord = coord[0]
                quotient_coord = coord[1] / coord[0]
            else:
                scale_coord = coord[1]
                quotient_coord = coord[0] / coord[1]
            # Evaluation of the scale_fct could be optimized
            # (since it only depends on the x-coordinate or only on
            # the y-coordinate, depending on the specific cone).
            if cone == 'r':
                cur_scale_fct = cur_hor_scale_fct
            elif cone == 't':
                cur_scale_fct = cur_vert_scale_fct
            elif cone == 'l':
                cur_scale_fct = cur_hor_scale_fct_flipped
            else:  # cone == 'b'
                cur_scale_fct = cur_vert_scale_fct_flipped

            # The following is slightly faster, but since the main execution
            # time is spent elsewhere, we refrain from using the optimized
            # (and untested) version.
            # if cone in hori_cones:
            #     # assert np.all(scale_coord[0] == scale_coord[1])
            #     scale_coord = scale_coord[0]
            # else:
            #     s = scale_coord.shape[0]
            #     scale_coord = scale_coord[:, 0].reshape((s, 1))
            #     # assert np.all(scale_coord[:, 0] == scale_coord[:, 1])
            # end of the new part

            spectrograms.append(cur_scale_fct.call(scale_coord) *
                                cur_dir_fct.call(quotient_coord))

        # calculate the "dual frame weights" and the frame bounds
        dual_frame_weight = np.zeros((self.height, self.width))
        for i, spect in enumerate(spectrograms):
            self._add_wrapped_to_matrix(i, np.square(spect), dual_frame_weight)
        self._frame_bounds = (np.min(dual_frame_weight),
                              np.max(dual_frame_weight))

        # ensure that the lower frame bound is not too small
        assert self._frame_bounds[0] >= 0.1

        if self.is_parseval:
            dual_frame_weight = np.sqrt(dual_frame_weight)
        normal_spects = []
        for i, spectro in enumerate(spectrograms):
            normal_spects.append(spectro / self._wrap(i, dual_frame_weight))
        if self.is_parseval:
            self._spectrograms = normal_spects
            self._frame_bounds = (1, 1)
        else:
            self.dual_spects = normal_spects
            self._spectrograms = spectrograms

    def _wrapped_to_index(self):
        return self._wrapped_to_index

    def _calculate_grids(self):
        values_x, values_y = self._xy_values()

        if self.periodization:
            (x_min, x_max) = (self.__x_min, self.__x_max)
            (y_min, y_max) = (self.__y_min, self.__y_max)
            (width, height) = (self.width, self.height)
            xy_values = [self._xy_values(x_min + i * width,
                                         x_max + i * width,
                                         y_min + j * height,
                                         y_max + j * height)
                         for i in range(-1, 2)
                         for j in range(-1, 2)]

            x_values = [value[0] for value in xy_values]
            y_values = [value[1] for value in xy_values]
            horizontal_quotient_grids = [div0(y_value, x_value)
                                         for x_value, y_value
                                         in zip(x_values, y_values)]
            vertical_quotient_grids = [div0(x_value, y_value)
                                       for x_value, y_value
                                       in zip(x_values, y_values)]
        else:
            x_values = [values_x]
            y_values = [values_y]

            horizontal_quotient_grids = [div0(values_y, values_x)]
            vertical_quotient_grids = [div0(values_x, values_y)]

        self._x_values = x_values
        self._y_values = y_values
        self._hor_quo_grids = horizontal_quotient_grids
        self._ver_quo_grids = vertical_quotient_grids
        self._low_pass_spect = (self.__horizontal_low_pass_fct.call(values_x) *
                                self.__vertical_low_pass_fct.call(values_y))

    def _calculate_full_spect(self, j, k, cone, unnormalized=False):
        # 'unnormalized' can be used to overwrite the "parseval" normalization.
        # This is important, since the parseval normalization requires the
        # dual_frame_weight, which can only be computed once all "unnormalized"
        # spectrograms are known!
        direction_function = self._mother_shearlet.direction_function
        cur_hor_scale_fct = MS.scale(self.__horizontal_scale_fct, 2**j)
        cur_vertical_scale_fct = MS.scale(self.__vertical_scale_fct, 2**j)
        dir_fct = MS.scale(MS.translate(direction_function, k),
                           2**((self._alphas[j] - 1) * j))

        cur_spect = np.zeros((self.height, self.width))

        for (x_value,
             y_value,
             hor_quo_grid,
             vert_quo_grid) in zip(self._x_values,
                                   self._y_values,
                                   self._hor_quo_grids,
                                   self._ver_quo_grids):
            if self.is_real:
                fac2 = np.zeros(hor_quo_grid.shape)
                if cone == 'h':
                    first_fact1 = cur_hor_scale_fct.call(x_value)
                    first_fact2 = cur_hor_scale_fct.call(-x_value)
                    fac1 = first_fact1 + first_fact2
                    fac2[:, fac1 > 0] = dir_fct.call(hor_quo_grid[:, fac1 > 0])
                elif cone == 'v':
                    first_fact1 = cur_vertical_scale_fct.call(y_value)
                    first_fact2 = cur_vertical_scale_fct.call(-y_value)
                    fac1 = first_fact1 + first_fact2
                    indices = (fac1 > 0).ravel()
                    fac2[indices, :] = dir_fct.call(vert_quo_grid[indices, :])

                ne.evaluate('cur_spect + fac1 * fac2', out=cur_spect)
                # cur_spect += fac2 * (first_fact1 + first_fact2)
                # spectrograms.append(fac2 *
                #                     (first_fact1 + first_fact2))
            else:
                fac2 = np.zeros(hor_quo_grid.shape)
                if cone == 'r':
                    fac1 = cur_hor_scale_fct.call(x_value)
                    # fac2 = dir_fct.call(hor_quo_grid)
                    fac2[:, fac1 > 0] = dir_fct.call(hor_quo_grid[:, fac1 > 0])
                elif cone == 't':
                    fac1 = cur_vertical_scale_fct.call(y_value)
                    indices = (fac1 > 0).ravel()
                    fac2[indices, :] = dir_fct.call(vert_quo_grid[indices, :])
                    # fac2 = dir_fct.call(vert_quo_grid)
                elif cone == 'l':
                    fac1 = cur_hor_scale_fct.call(-x_value)
                    fac2[:, fac1 > 0] = dir_fct.call(hor_quo_grid[:, fac1 > 0])
                    # fac2 = dir_fct.call(hor_quo_grid)
                elif cone == 'b':
                    fac1 = cur_vertical_scale_fct.call(-y_value)
                    assert fac1 is not None  # silence pyflakes
                    indices = (fac1 > 0).ravel()
                    fac2[indices, :] = dir_fct.call(vert_quo_grid[indices, :])
                    # fac2 = dir_fct.call(vert_quo_grid)
                    assert fac2 is not None  # silence pyflakes

                ne.evaluate('cur_spect + fac1 * fac2',
                            out=cur_spect)
                # cur_spect += fac_1 * fac2
                # spectrograms.append(fac_1 * fac2)

        if unnormalized:
            return cur_spect
        else:
            if self.is_parseval:
                return cur_spect / self.dual_frame_weight
            else:
                return cur_spect

    def _calculate_full_spectrograms(self, verbose):
        if not verbose:
            my_tqdm = lambda x, desc: x
        else:
            my_tqdm = tqdm

        self._calculate_grids()

        if not self._generator:
            # add low-pass spectrogram (this does not require periodization)
            spectrograms = [self._low_pass_spect]

            # compute remaining spectrograms

            for j, k, cone in my_tqdm(self._indices[1:],
                                      desc="Precomputing shearlet system"):
                cur_spect = self._calculate_full_spect(j, k, cone,
                                                       unnormalized=True)
                spectrograms.append(cur_spect)

            # in case of a parseval frame
            # -> do the normalization that was supressed above
            if self.is_parseval:
                # although the current (non-normalized) spectrograms are
                # not the final ones, this assignment is needed to compute
                # the correct 'dual frame weights'
                self._spectrograms = spectrograms
                dual_frame_weight = self.dual_frame_weight

                for i in range(len(spectrograms)):
                    spectrograms[i] /= dual_frame_weight

            self._spectrograms = spectrograms

    def _spectrograms_generator(self):
        assert self._generator, ("This method should only be used for the "
                                 "'generator' type alpha shearlet transform!")
        yield self._low_pass_spect

        for j, k, cone in self._indices[1:]:
            yield self._calculate_full_spect(j, k, cone)

    @property
    def spectrograms(self):
        r"""
        A list of the spectrograms (Fourier transforms) of the alpha-shearlets
        associated to the transform.

        .. note::
            If the transform was constructed with ``generator=True``, the
            return value is a *generator* instead of a list.

        The ordering of this list is the same as that of :func:`indices` and
        of the return value of :func:`transform`, i.e., if
        ``coeff = self.transform(im)`` for some image ``im``, then ``coeff[i]``
        are the coefficients associated to the alpha-shearlet with Fourier
        transform ``self.spectrograms[i]`` whose "geometric meaning"
        ``(scale, shear, cone)`` is given by ``self.indices[i]``.

        .. warning::
            The spectrograms are optimized for plotting, via
            :func:`matplotlib.pyplot.imshow`, not for computations. Hence, they
            are still "FFT shifted". This can be undone by considering
            ``fourier_util.my_ifft_shift(self.spectrograms[i])``.
        """
        if not self._generator:
            return self._spectrograms
        else:
            return self._spectrograms_generator()

    @property
    def shearlets(self):
        r"""
        A list (or a generator) of the (space-side, non-normalized)
        alpha-shearlets associated to the transform.

        A generator is returned if the ``self`` object was created with the
        option ``generator=True``.
        """
        im = np.zeros((self.height, self.width))
        im[self.height // 2, self.width // 2] = 1
        if self.is_real:
            return np.real(self.transform(im, do_norm=False))
        else:
            return self.transform(im, do_norm=False)

    @classmethod
    def calculate_indices(cls, real, alphas):
        r"""
        This function can be used to compute the *indices* associated to an
        alpha-shearlet system with the given properties.

        Given these indices, one can then compute e.g. the number of directions
        per scale and the total number of shearlets. For more details on what
        these *indices* are, cf. :func:`indices`.

        .. note:: One can get the same result by creating an object
                  ``my_trafo`` of the class :class:`AlphaShearletTransform`
                  and then invoke `my_trafo.indices`. However, using the
                  ``calculate_indices`` method is much faster.

        :param bool real:
            Flag determining whether the indices should be computed for an
            alpha-shearlet system consisting of real-valued alpha-shearlets,
            or not. See the documentation of :class:`AlphaShearletTransform`
            for more details.

        :param list alphas:
            List of alpha values which determines both, the number of scales
            and the value of alpha on each scale. See the documentation of
            :class:`AlphaShearletTransform` for more details.

        :return:
            (:class:`list`) -- A list of indices associated to an
            alpha-shearlet system with the given parameters.
        """
        if real:
            cones = ['h', 'v']
        else:
            cones = ['r', 't', 'l', 'b']
        indices = [-1]  # -1 is the low-pass filter
        for j, alpha in enumerate(alphas):
            for cone in cones:
                # k = int(round(2**((1 - alpha) * j)))
                k = math.ceil(2**((1 - alpha) * j))

                # to get a nice visual appearance, we change
                # the ordering of the shears on certain cones.
                shear_range = range(-k, k + 1)
                if cone in ['t', 'b', 'v']:
                    shear_range = range(k, -k - 1, -1)

                for shear in shear_range:
                    indices.append((j, shear, cone))
        return indices

    @classmethod
    def num_shears(cls, alphas, j, real=False):
        if j == -1:
            return 1
        # shears_per_cone = 2 * int(round(2**((1 - alphas[j]) * j))) + 1
        shears_per_cone = 2 * math.ceil(2**((1 - alphas[j]) * j)) + 1
        if not real:
            return 4 * shears_per_cone
        else:
            return 2 * shears_per_cone

    def _get_do_norm_frame_bounds(self):
        fourier_sum = np.zeros((self.height, self.width))
        if not self.is_subsampled:
            for spect, norm in zip(self.spectrograms, self.space_norms):
                fourier_sum += np.square(spect / norm)
        else:
            for i, (spect, norm) in enumerate(zip(self.spectrograms,
                                                  self.space_norms)):
                self._add_wrapped_to_matrix(i,
                                            np.square(spect / norm),
                                            fourier_sum)
        return (np.min(fourier_sum), np.max(fourier_sum))

    def get_frame_bounds(self, do_norm=False):
        r"""
        This method yields the *frame bounds* of the associated transform as
        a tuple ``(A,B)``, where ``A`` and ``B`` are the lower and upper frame
        bounds, respectively.

        Hence, if ``(A, B) = self.get_frame_bounds(do_norm=b)`` and
        ``M = numpy.linalg.norm(im) ** 2``, then::

            A * M <= numpy.linalg.norm(self.transform(im, do_norm=b)) <= B*M

        for all images ``im``.

        :param bool do_norm:
            This is the same flag as passed to :func:`transform`. Recall that
            in case of ``do_norm=False``, the transform calculates the
            convolutions :math:`f \ast \psi_i`, while for ``do_norm=True``, it
            calculates the convolutions with respect to the *normalized*
            alpha-shearlet filters, i.e.,
            :math:`f \ast \frac{\psi_i}{\| \psi_i \|_{L^2}}`.

            .. note:: The transform is written in such a way that a frame with
                      good frame bounds is obtained for ``do_norm=False``.
                      Setting ``do_norm=True`` will result in much worse frame
                      bounds. Recall, however, that the one frame is obtained
                      by a simple rescaling of a frame with good bounds.
        """
        if not do_norm:
            if self.is_parseval:
                return (1, 1)
            else:
                if self._frame_bounds is None:
                    # for the subsampled transform, the frame bounds must be
                    # calculated in the constructor
                    assert not self.is_subsampled

                    self._frame_bounds = (np.min(self.dual_frame_weight),
                                          np.max(self.dual_frame_weight))
                return self._frame_bounds
        else:
            return self._get_do_norm_frame_bounds()

    @property
    def indices(self):
        r"""
        This property yields a list of so-called *indices* which describe
        the geometric meaning of the different "parts" of the alpha-shearlet
        coefficients.

        To better explain this, consider the following example::

            >>> my_trafo = AlphaShearletTransform(512, 512, [0.5]*2)
            >>> my_trafo.indices
            [-1,
             (0, -1, 'r'), (0, 0, 'r'), (0, 1, 'r'),
             (0, 1, 't'), (0, 0, 't'), (0, -1, 't'),
             (0, -1, 'l'), (0, 0, 'l'), (0, 1, 'l'),
             (0, 1, 'b'), (0, 0, 'b'), (0, -1, 'b'),
             (1, -2, 'r'), (1, -1, 'r'), (1, 0, 'r'), (1, 1, 'r'), (1, 2, 'r'),
             (1, 2, 't'), (1, 1, 't'), (1, 0, 't'), (1, -1, 't'), (1, -2, 't'),
             (1, -2, 'l'), (1, -1, 'l'), (1, 0, 'l'), (1, 1, 'l'), (1, 2, 'l'),
             (1, 2, 'b'), (1, 1, 'b'), (1, 0, 'b'), (1, -1, 'b'), (1, -2, 'b')]
            >>> im = numpy.random.random((512,512))
            >>> coeff = my_trafo.transform(im)
            >>> coeff.shape
            (33, 512, 512)
            >>> len(my_trafo.indices)
            33

        First of all, note that the first dimension of ``coeff`` is the same as
        the length of ``my_trafo.indices``. Precisely, ``my_trafo.indices[i]``
        encodes information about the alpha-shearlet :math:`\psi_i` which is
        used to compute :math:`\mathrm{coeff}[i] = \mathrm{im} \ast \psi_i`.

        The special index ``-1`` means that the associated alpha-shearlet
        belongs to the low-pass part.

        All other indices are of the form ``(j, k, c)``, where this
        3-tuple describes the properties of the associated alpha-shearlet
        :math:`\psi_i`. Precisely,

        * ``j`` encodes the *scale* (between 0 and ``self.num_scales - 1``)
          of :math:`\psi_i`.

        * ``k`` encodes the amount of *shearing*. This is always an integer
          satisfying :math:`-\lceil 2^{j(1-\alpha)} \rceil \leq k \leq
          \lceil 2^{j(1-\alpha)} \rceil`,
          where ``alpha`` denotes the value of alpha associated to scale ``j``.

        * ``c`` encodes the *frequency cone* in which the alpha-shearlet
          :math:`\psi_i` has its frequency support. For the case of
          complex-valued alpha-shearlets, the frequency plane is divided into
          the following four cones:

              +------------+-------+-------+------+--------+
              | value of c |  'r'  |  't'  | 'l'  |  'b'   |
              +============+=======+=======+======+========+
              |    cone    | right |  top  | left | bottom |
              +------------+-------+-------+------+--------+

          For the case of real-valued alpha-shearlets, there are just two
          frequency cones, the *horizontal* one (encoded by ``'h'``) and
          the *vertical* one (encoded by ``'v'``).

        To get an intuitive understanding of the meaning of the indices and
        to understand the ordering of the different alpha-shearlets, the
        following code snippet might be helpful. It loops over all shearlets,
        prints their index and displays a plot of the spectrogram of the
        shearlet::

            >>> import matplotlib.pyplot as plt
            >>> from AlphaTransform import AlphaShearletTransform as AST
            >>> my_trafo = AST(512, 512, [0.5]*3)
            >>> for index, spect in zip(my_trafo.indices,
                                        my_trafo.spectrograms):
                    plt.imshow(spect)
                    plt.title("index: {0}".format(index))
                    plt.show()

        One might also experiment with the additional arguments ``real=True``
        or ``periodization=False`` (passed to the ``AST`` constructor) to see
        their effect.
        """
        return self._indices

    def scale_slice(self, scale):
        r"""
        Convenience method to determine, for a given scale, the part of the
        spectrograms or of the transform coefficients that belong to the given
        scale.

        As an example, consider the following::

            >>> from AlphaTransform import AlphaShearletTransform
            >>> my_trafo = AlphaShearletTransform(512, 512, [0.5]*3)
            >>> my_trafo.scale_slice(-1)
            slice(0, 1, None)
            >>> my_trafo.scale_slice(0)
            slice(1, 13, None)
            >>> my_trafo.scale_slice(1)
            slice(13, 33, None)
            >>> my_trafo.scale_slice(2)
            slice(33, 53, None)

        Hence, if ``coeff = my_trafo.transform(im)`` for a given image ``im``,
        then ``coeff[0:1]`` is the part belonging to scale ``-1``
        (i.e., the low-pass part), ``coeff[1:13]`` is the part associated to
        scale ``0``, etc.

        One can even write ``coeff[my_trafo.scale_slice(1)]`` to directly
        get the part of the transform coefficients associated to scale ``1``,
        etc.
        """
        assert -1 <= scale < self.num_scales, ("The given scale is outside "
                                               "the valid range")
        if scale == -1:
            return slice(0, 1)
        lower_bound = next(i
                           for i, index in enumerate(self.indices)
                           if index != -1 and index[0] == scale)
        upper_bound = (len(self.indices) if scale + 1 == self.num_scales
                       else next(i
                                 for i, index in enumerate(self.indices)
                                 if index != -1 and index[0] == scale + 1))
        return slice(lower_bound, upper_bound)

    # implementation decision: for even grid sizes (there is no 'middle'),
    # put 'additional' point left-most (bottom-most), not right-most (top-most)
    # Note: the grid ranges (in x direction) from -n-epsilon_width, ..., n.
    #       the indices range (in x direction) from 0, ..., 2n+epsilon_width
    def _calculate_bounds(self):
        epsilon_width = 1 - (self.width % 2)
        # Note: the result is always an integer,
        # but python does not know this, so we use // division
        n = (self.width - 1 - epsilon_width) // 2
        assert self.width == 2 * n + 1 + epsilon_width

        epsilon_height = 1 - (self.height % 2)
        m = (self.height - 1 - epsilon_height) // 2
        assert self.height == 2 * m + 1 + epsilon_height

        self.__x_min = -n - epsilon_width
        self.__x_max = n
        self.__y_min = -m - epsilon_height
        self.__y_max = m

    def _xy_values(self, x_min=None, x_max=None, y_min=None, y_max=None):
        x_min = self.__x_min if x_min is None else x_min
        y_min = self.__y_min if y_min is None else y_min
        x_max = self.__x_max if x_max is None else x_max
        y_max = self.__y_max if y_max is None else y_max
        x_values = np.mgrid[x_min:x_max + 1]
        y_values = np.mgrid[y_max:y_min - 1:-1].reshape((y_max - y_min + 1, 1))
        return (x_values, y_values)

    def _grid(self, x_min=None, x_max=None, y_min=None, y_max=None):
        # NOTE : in numpy, the y coordinates begin on the top with low values.
        # We want high values to be on top (as in mathematics),
        # so we use a stride of -1
        x_min = self.__x_min if x_min is None else x_min
        y_min = self.__y_min if y_min is None else y_min
        x_max = self.__x_max if x_max is None else x_max
        y_max = self.__y_max if y_max is None else y_max
        grid_temp = np.mgrid[y_max:y_min - 1:-1,
                             x_min:x_max + 1]
        return (grid_temp[1], grid_temp[0])

    def _rescale_filters(self):
        # calculate (horizontal and vertical) scaling factors
        n_max = - self.__x_min
        m_max = - self.__y_min

        scale_function = self._mother_shearlet.scale_function
        scale_fun_lower_bound = scale_function.large_support[0]
        scale_fun_upper_bound = scale_function.large_support[1]

        max_scale = 2 ** (-(self.num_scales - 1))

        R = min(n_max, m_max)
        # this is the size of the (quadratic(!)) low pass region
        a = max_scale * scale_fun_lower_bound / scale_fun_upper_bound * R
        assert a >= 4, ("The given number of scales is too large "
                        "for the given dimensions!")

        scale_nominator = ((scale_fun_upper_bound - scale_fun_lower_bound) /
                           max_scale)
        hor_scale = scale_nominator / (n_max - a / max_scale)
        vert_scale = scale_nominator / (m_max - a / max_scale)
        hor_shift = hor_scale * a - scale_fun_lower_bound
        vert_shift = vert_scale * a - scale_fun_lower_bound

        self.__horizontal_scale_fct = MS.scale(MS.translate(scale_function,
                                                            hor_shift),
                                               1 / hor_scale)
        self.__vertical_scale_fct = MS.scale(MS.translate(scale_function,
                                                          vert_shift),
                                             1 / vert_scale)

        # horizontal_scale = n_max / orig_scale_fun_upper_bound * max_scale
        # vertical_scale = m_max / orig_scale_fun_upper_bound * max_scale

        # self.__horizontal_scale_fct = MS.scale(orig_scale_function,
        #                                        horizontal_scale)
        # self.__vertical_scale_fct = MS.scale(orig_scale_function,
        #                                      vertical_scale)

        # rescale the low-pass functions
        orig_low_pass_function = self._mother_shearlet.low_pass_function
        # hor_scale_fun_low_bound =self.__horizontal_scale_fct.large_support[0]
        # hori_scale_low_pass = (hor_scale_fun_low_bound /
        #                        orig_low_pass_function.large_support[1])
        hori_scale_low_pass = (a /
                               orig_low_pass_function.large_support[1])
        self.__horizontal_low_pass_fct = MS.scale(orig_low_pass_function,
                                                  hori_scale_low_pass)

        # vert_scale_fun_low_bound = self.__vertical_scale_fct.large_support[0]
        # vert_scale_low_pass = (vert_scale_fun_low_bound /
        #                        orig_low_pass_function.large_support[1])
        # self.__vertical_low_pass_fct = MS.scale(orig_low_pass_function,
        #                                         vert_scale_low_pass)
        vert_scale_low_pass = (a /
                               orig_low_pass_function.large_support[1])
        self.__vertical_low_pass_fct = MS.scale(orig_low_pass_function,
                                                vert_scale_low_pass)

    @property
    def periodization(self):
        r"""
        Boolean flag indicating whether the spectrograms of the alpha-shearlets
        which exceed the frequency range of the discrete fourier transform are
        periodized (value ``True``) or truncated (value ``False``).

        This is the same as the value of the parameter ``periodization`` that
        was passed to the constructor of :class:`AlphaShearletTransform` to
        construct ``self``.

        To see the difference, executing the following code with different
        values for ``periodization`` might be helpful::

            >>> from AlphaTransform import AlphaShearletTransform
            >>> import matplotlib.pyplot as plt
            >>> periodization = True
            >>> my_trafo = AlphaShearletTransform(512, 512, [0.5]*3,
                                                  periodization=periodization)
            >>> i = my_trafo.indices.index((2, -2, 'r'))
            >>> plt.imshow(my_trafo.spectrograms[i])
            >>> plt.show()
        """
        return self._periodization

    @property
    def num_scales(self):
        r"""
        The number of scales of the alpha-shearlet system. This is the same
        as the length of the parameter ``alphas`` that was passed to the
        constructor of :class:`AlphaShearletTransform` to construct ``self``.
        """
        # return self.__num_scales
        return len(self._alphas)

    @property
    def width(self):
        r"""
        This property yields the width (in pixels, as an :class:`int`) of the
        images that can be analyzed using the ``self`` object.
        """
        return self._width

    @property
    def height(self):
        r"""
        This property yields the height (in pixels, as an :class:`int`) of the
        images that can be analyzed using the ``self`` object.
        """
        return self._height

    @property
    def is_subsampled(self):
        r"""
        Boolean flag indicating whether the transform is subsampled. This is
        just the same as the value of ``subsampled`` that was passed to the
        constructor of :class:`AlphaShearletTransform` to construct ``self``.
        """
        return self._subsampled

    @property
    def is_parseval(self):
        r"""
        Boolean flag indicating whether the transform was normalized to
        obtain a Parseval frame. This is just the same as the value of
        ``parseval`` that was passed to the constructor of
        :class:`AlphaShearletTransform` to constructo ``self``.
        """
        return self._parseval

    @property
    def is_real(self):
        r"""
        Boolean flag indicating whether the transform was symmetrized in
        Fourier to obtain *real-valued* alpha-shearlets. This is just the same
        as the value of ``real`` that was passed to the constructor of
        :class:`AlphaShearletTransform` to construct ``self``.
        """
        return self._real

    @property
    def fourier_norms(self):
        r"""
        This property yields the *Fourier-side* L² norms of the individual
        spectrograms of the transform (as a tuple(!)).

        To obtain the (usually more important) *space-side* L² norms, use
        the property :func:`space_norms` instead.

        :rtype: tuple of floats
        """
        self._create_cache()
        return self._fourier_norms

    @property
    def space_norms(self):
        r"""
        This property yields the *space-side* L² norms of the analyzing
        alpha-shearlets. The return value is a single tuple containing
        all norms.

        .. note::
            We use the unitary version of the Fourier transform, so that
            the dirac delta :math:`\delta_0` at the origin satisfies
            :math:`\mathcal{F}[\delta_0]
            \equiv 1 / \sqrt{\mathrm{width} \cdot \mathrm{height}}`.

            Regarding the convolution theorem, this implies

            .. math::

                f \ast g &= \sqrt{\mathrm{width} \cdot \mathrm{height}} \cdot
                            \mathcal{F}^{-1}[\hat{f} \cdot \hat{g}] \\
                         &= \mathcal{F}^{-1}([\sqrt{\mathrm{width}
                                              \cdot \mathrm{height}}
                                              \cdot \hat{f}]
                                             \cdot \hat{g}).

            Hence, since we implement the convolution with the different
            alpha-shearlets as multiplication in the Fourier domain with the
            elements of ``self.spectrograms``, the i-th space-side shearlet
            is given by

            .. math::
                \psi_i = \mathcal{F}^{-1}[\sqrt{\mathrm{width}
                                                \cdot \mathrm{height}}
                                          \cdot \mathrm{self.spectrograms[i]}].

            Hence, its L² norm is
            :math:`\|\psi_i\|_{L^2}
            = \sqrt{\mathrm{width} \cdot \mathrm{height}}
            \cdot \| \mathrm{self.spectrograms[i]} \|_{L^2}`
        """
        self._create_cache()
        return self._space_norms

    @property
    def redundancy(self):
        r"""
        The redundancy of the alpha-shearlet frame associated to ``self``.
        This is the number of coefficients of the transform divided by the
        number of coefficients of the input image (i.e., ``width * height``).

        For the fully sampled transform, associated to each alpha-shearlet,
        the number of coefficients is identical to the dimension of the input.
        Hence, the redundancy is the same as the total number of
        alpha-shearlets, i.e., ``len(self.indices)``.

        For the subsampled transform, the redundancy is much lower, but not
        as easy to calculate. Hence, this property is useful.
        """
        if self.is_subsampled:
            dimension = sum([spec.shape[0] * spec.shape[1]
                             for spec in self.spectrograms])
            return dimension / (self.height * self.width)

        else:
            return len(self.indices)
            # return len(self.spectrograms)

    def alpha(self, scale):
        """
        Return the value of alpha for the given scale.

        :param int scale:
            The scale for which the value of alpha should be returned.
            Must satisfy ``0 <= scale <= self.num_scales``.
            Typically, ``scale`` is the first element of an *index* tuple,
            cf. :func:`indices`.

        :return:
            (:class:`float`)
            The value of alpha associated to scale ``scale``.
            We always have ``0 <= self.alpha(scale) <= 1``.
        """
        return self._alphas[scale]

    def _add_wrapped_to_matrix(self, i, source, target):
        r"""
        Add the 'wrapped' Fourier-side object 'source' to the matrix 'target'.
        The exact type of wrapping is determined by 'i'.

        A typical use case of this method is to add a (wrapped) spectrogram,
        or a (Fourier-side) transform to 'target'. Because of the "wrapping"
        (which is used to achieve subsampling (in case of
        self.is_subsampled == True), this is nontrivial without using this
        method.

        Parameters:
            i: This is an integer in [0, ..., len(self.spectrograms)].
               It determines the exact kind of 'dewrapping' to be used.

            source: numpy.array object. This is the 'Fourier-side' object which
                    is to be added (in "un-wrapped" form) to 'target'.

                    Typically, 'source' will be the i-th element of the return
                    value of AlphaShearletTransform._transform_fourier,
                    or the i-th element of self.spectrograms, etc.

            target: numpy.array object with shape (self.height, self.width),
                    to which 'source' is added.
        """
        if self._subsampled:
            if i != 0:
                # as usual: y-index needs to be the first, x-index the second
                np.add.at(target,
                          tuple(self._wrapped_to_index[i][::-1]),
                          source)
            else:
                (x_min, x_max) = self._wrapped_to_index[0][0]
                (y_min, y_max) = self._wrapped_to_index[0][1]
                # print(x_min, x_max, y_min, y_max)
                # print(x_max - x_min, y_max - y_min)
                # print(source.shape)
                # print(target.shape)
                target[y_min: y_max + 1, x_min: x_max + 1] += source
        else:
            target += source

    def _wrap(self, i, source):
        r"""
        Return the wrapped/periodized part of the fourier-side object 'source'.
        The exact slice and periodization/wrapping to be used is determined
        by the index 'i'.

        Parameters:
            i: This is an integer in [0, ..., len(self.spectrograms)].
               It determines the exact kind of 'wrapping' to be used.

            source: numpy.array object. This is the 'Fourier-side' object which
                    is to be wrapped.

                    Note: The dimensions of 'source' should be
                          (self.height, self.width).

        Returns:
            The 'wrapped' version of 'source'.
        """
        assert self.is_subsampled
        if i != 0:
            # as usual: y-index needs to be first, x-index second
            return source[self._wrapped_to_index[i][::-1]]
        else:
            (x_min, x_max) = self._wrapped_to_index[0][0]
            (y_min, y_max) = self._wrapped_to_index[0][1]
            return source[y_min: y_max + 1, x_min: x_max + 1]

    def _unwrap(self, i, wrapped_spect):
        r"""
        Unwrap the (wrapped) spectrogram 'wrapped_spect'.

        The type of wrapping is determined by 'i',
        see also AlphaShearletTransform._add_wrapped_to_matrix.

        A typical use case of this method is to unwrap a (wrapped) spectrogram,
        or a (Fourier-side) transform.

        Note: For efficiency reasons, this method should NOT be used for
              computations, only for visualization (mainly for debugging)

        Parameters:
            i: This is an integer in [0, ..., len(self.spectrograms)].
               It determines the exact kind of 'dewrapping' to be used.

            wrapped_spect: numpy.array object. This is the 'Fourier-side'
                           object which is to be "unwrapped".

                           Typically, wrapped_spect will be the i-th element
                           of the "transform" list returned by
                           AlphaShearletTransform._transform_fourier,
                           or the i-th element of self.spectrograms, etc.
        """
        result = np.zeros((self._height, self._width))
        self._add_wrapped_to_matrix(i, wrapped_spect, result)
        return result

    def _transform_fourier(self, im_fourier):
        """
        Computes the (Fourier-side) alpha transform of 'im_fourier',
        with parameters of the alpha transform determined by the 'self' object.

        Parameters:
            im_fourier : The 'my_fft_shift'ed Fourier transform of the image

        Returns:
            A list (or a numpy array), consisting of 'im_fourier'
            multiplied with each of the spectograms of the alpha transform,
            suitably truncated and wrapped, if self.subsampled is True.

            Note: The 'pieces' of the transform are still 'my_fft_shift'ed.
        """
        if self.is_subsampled:
            trafos = []
            # calculate low-pass part
            (x_min, x_max) = self._wrapped_to_index[0][0]
            (y_min, y_max) = self._wrapped_to_index[0][1]
            # NOTE : first index is row (y-coord.) and second is x-coord.
            low_pass_part = self.spectrograms[0] * im_fourier[y_min: y_max + 1,
                                                              x_min: x_max + 1]
            trafos.append(low_pass_part)
            # calculate remaining parts of the transform
            for spect, indices in zip(self.spectrograms[1:],
                                      self._wrapped_to_index[1:]):
                # NOTE : first index is row (y-coord.) and second is x-coord.
                trafos.append(im_fourier[indices[1], indices[0]] * spect)
            return trafos
        else:
            # Originally, this returned a list, not a numpy.array!
            return np.array([im_fourier * spect
                             for spect in self.spectrograms])

    def _transform_fourier_generator(self, im_fourier):
        if self.is_subsampled:
            # calculate low-pass part
            (x_min, x_max) = self._wrapped_to_index[0][0]
            (y_min, y_max) = self._wrapped_to_index[0][1]
            # NOTE : first index is row (y-coord.) and second is x-coord.
            low_pass_part = self.spectrograms[0] * im_fourier[y_min: y_max + 1,
                                                              x_min: x_max + 1]
            yield low_pass_part
            # calculate remaining parts of the transform
            for spect, indices in zip(self.spectrograms[1:],
                                      self._wrapped_to_index[1:]):
                # NOTE : first index is row (y-coord.) and second is x-coord.
                yield im_fourier[indices[1], indices[0]] * spect
        else:
            for spect in self.spectrograms:
                yield im_fourier * spect

    def transform(self, image, do_norm=True):
        r"""
        Computes the alpha-shearlet transform of ``image``, where the
        properties (number of scales, value of alpha, etc.) of the
        transform are determined by the ``self`` object.

        **Parameters**

        :param numpy.ndarray image:
            The "image" of which the transform should be computed.
            ``image`` must be a 2-dimensional :class:`numpy.ndarray` satisfying
            ``image.shape == (self.height, self.width)``.

        :param bool do_norm:
            Boolean flag indicating whether the transform should be
            normalized by the (space-side) L² norms of the analyzing
            alpha-shearlets. Hence, if ``do_norm=True``, then the returned
            coefficients correspond to
            :math:`\mathrm{image} \ast \psi_i / \|\psi_i\|_{L^2}`, instead
            of :math:`\mathrm{image} \ast \psi_i`.

        **Return value**

        :returns:
            The (possibly normalized) alpha-shearlet coefficients of ``image``.
            For a precise description of what is computed, we refer to the
            technical report.

            Depending on the settings of the ``self`` object, the *return type*
            varies:

            * For the fully sampled transform with ``generator=False``, the
              return value is a single 3-dimensional :class:`numpy.ndarray`
              of dimension ``(self.redundancy, self.height, self.width``).

            * For the fully sampled transform with ``generator=True``, the
              return value is a *generator* which produces the same output
              (one 2-dimensional :class:`numpy.ndarray` at a time) as for
              the ordinary fully sampled transform.

            * For the subsampled transform, the return value is a *list* of
              2-dimensional numpy arrays **of varying dimensions**.
        """
        if self._generator:
            return self.transform_generator(image, do_norm)

        assert (self.width == image.shape[1] and
                self.height == image.shape[0]), ("Dimensions of the "
                                                 "image ({0}) do not match "
                                                 "those of the "
                                                 "transform ({1}).").format(
                                                     image.shape,
                                                     (self.height, self.width))
        image_fourier = my_fft_shift(self._fft(-1, image))

        trafo = self._transform_fourier(image_fourier)
        if do_norm:
            for i, norm in enumerate(self.space_norms):
                trafo[i] = self._ifft(i, my_ifft_shift(trafo[i])) / norm
        else:
            for i in range(len(trafo)):
                trafo[i] = self._ifft(i, my_ifft_shift(trafo[i]))
        return trafo

    def transform_generator(self, image, do_norm=True):
        r"""
        This method does the same as :func:`transform`, but always returns a
        *generator* instead of a :class:`list` or a :class:`numpy.ndarray`.
        """
        assert (self.width == image.shape[1] and
                self.height == image.shape[0]), ("Dimensions of the "
                                                 "image ({0}) do not match "
                                                 "those of the "
                                                 "transform ({1}).").format(
                                                     image.shape,
                                                     (self.height, self.width))
        image_fourier = my_fft_shift(self._fft(-1, image))
        trafo_fourier = self._transform_fourier_generator(image_fourier)
        if not do_norm:
            for i, trafo in enumerate(trafo_fourier):
                yield self._ifft(i, my_ifft_shift(trafo))
        else:
            for i, (norm, trafo) in enumerate(zip(self.space_norms,
                                                  trafo_fourier)):
                yield self._ifft(i, my_ifft_shift(trafo)) / norm

    def adjoint_transform(self, coeffs, spectrograms=None, do_norm=True):
        r"""
        This method computes and returns the adjoint operator to the operator
        computed by the method :func:`transform`. Since the :func:`transform`
        method is the *analysis operator* associated to the shearlet system,
        this means that :func:`adjoint_transform` is the associated *synthesis
        operator*.

        .. note:: The main use case of this method is if the transform is used
                  in convex programming. Most solvers require that the linear
                  transforms provide a method to apply the linear operator, as
                  well as the adjoint of the linear operator.

        **Required parameters**

        :param coeffs:
            This should be roughly of the same type as the return value of
            :func:`transform`. Usually, this argument will even be obtained
            directly or indirectly from :func:`transform`, e.g. like the
            following::

                # initialize the image 'im' and the transform object 'my_trafo'
                ... # omitted for brevity
                coeff = my_trafo.transform(im)
                mod_coeff = coeff * (np.abs(coeff) > 0.5)
                im2 = my_trafo.adjoint_transform(mod_coeff)

            .. note:: It is not necessary that the type of ``coeffs`` is
                      precisely the same as that of the return value of
                      :func:`transform`. For example, even if :func:`transform`
                      returns a 3-dimensional numpy array, it is possible for
                      ``coeffs`` to be a generator or a list. The crucial point
                      is that the *dimensions match*, i.e., that the following
                      code runs through::

                          coeff_test = self.transform(im)
                          for c, c_t in zip(coeff, coeff_test):
                              assert c.shape == c_t.shape

        **Keyword parameters**

        :param spectrograms:
            This parameter can be used to determine another system for
            synthesis than the shearlet system. If the default value (``None``)
            is passed, the synthesis is done with respect to the shearlet
            system associated to the ``self`` object.

            .. warning:: The parameter ``spectrograms`` is mainly for internal
                         use. Only use it if you know what you are doing.

        :param bool do_norm:
            If one wants to compute the adjoint operator to the
            :func:`transform` method, then the value of ``do_norm`` has to be
            the same as for the invocation of :func:`transform`. The default
            value is ``True`` in both cases.
            Cf. the documentation of :func:`transform` for more details.

        **Return value**

        :returns: (:class:`numpy.ndarray`) -- The result of the synthesis
                  operator (associated to the (alpha-shearlet) system
                  determined by ``spectrograms``) applied to ``coeff``.
        """
        if spectrograms is None:
            spectrograms = self.spectrograms

        # Maybe there is a more reliable way
        # to determine the correct 'dtype'?
        # result = np.zeros((self.height, self.width), dtype=coeffs[0].dtype)
        result = np.zeros((self.height, self.width), dtype='complex128')

        if self.is_subsampled:
            for i, (norm, coeff, spect) in enumerate(zip(self.space_norms,
                                                         coeffs,
                                                         spectrograms)):
                # adj_coeff = my_fft_shift(fft2(fftshift(coeffs[i])))
                # adj_coeff = my_fft_shift(fft2(coeffs[i]))

                if do_norm:
                    coeff = coeff / norm

                adj_coeff = my_fft_shift(self._fft(i, coeff))
                self._add_wrapped_to_matrix(i,
                                            adj_coeff * spect,
                                            result)
        else:
            for norm, coeff, spect in zip(self.space_norms,
                                          coeffs,
                                          spectrograms):
                # result += spect * my_fft_shift(fft2(fftshift(coeff)))
                # result += spect * my_fft_shift(fft2(coeff))
                if do_norm:
                    result += ((spect / norm) *
                               my_fft_shift(self._fft(-1, coeff)))
                else:
                    result += spect * my_fft_shift(self._fft(-1, coeff))

        return ifft2(my_ifft_shift(result))

    def _create_cache(self):
        # This function precomputes the following quantities:
        #    * self._dual_frame_weight
        #    * self._space_norms
        #    * self._fourier_norms.
        # Especially for the "generator version" of the transform, this is
        # much more efficient, since a single pass over all spectrograms
        # suffices to compute all of these quantities.
        if self._cache:
            return

        if not self.is_subsampled:
            # if fully sampled, we do compute self._dual_frame_weight
            self._dual_frame_weight = np.zeros((self.height, self.width))
            self._fourier_norms = []
            self._space_norms = []
            for spect in self.spectrograms:
                self._dual_frame_weight += np.square(spect)
                spect_norm = np.linalg.norm(spect)
                self._fourier_norms.append(spect_norm)
                self._space_norms.append(spect_norm / math.sqrt(spect.size))

            self._fourier_norms = tuple(self._fourier_norms)
            self._space_norms = tuple(self._space_norms)
            if self.is_parseval:
                self._dual_frame_weight = np.sqrt(self._dual_frame_weight)
        else:
            # if subsampled, self._dual_frame_weight is not needed
            self._fourier_norms = []
            self._space_norms = []
            for spect in self.spectrograms:
                spect_norm = np.linalg.norm(spect)
                self._fourier_norms.append(spect_norm)
                self._space_norms.append(spect_norm / math.sqrt(spect.size))

            self._fourier_norms = tuple(self._fourier_norms)
            self._space_norms = tuple(self._space_norms)

        self._cache = True

    @property
    def dual_frame_weight(self):
        r"""
        For the fully sampled transform, the *dual frame*
        :math:`(\tilde{\psi_i})_i` is given by
        :math:`\tilde{\psi_i} = \mathcal{F}^{-1}[\psi_i / w]`,
        where :math:`(\psi_i)_i` is the alpha-shearlet frame.
        The weight ``w``, which is called the *dual frame weight*,
        is given by this property.

        :return: (2-dimensional :class:`numpy.ndarray`) The dual frame weight w
        """
        assert not self.is_subsampled, ("The dual frame weights should only "
                                        "be used for the fully sampled "
                                        "transform!")
        self._create_cache()
        return self._dual_frame_weight

    def _calculate_inverse_spectrograms(self):
        # pylint: disable=unused-variable
        # since we are using ne.evaluate, so that the uses of some of the
        # variables are 'hidden' inside a string
        if self.is_subsampled:
            # for the subsampled transform, this method is not really needed,
            # unless one wants to (e.g.) plot the inverse spectrograms.
            return self.dual_spects
            # return None
        if self.is_parseval:
            return self.spectrograms

        # dual_weight = self.__dual_frame_weight
        dual_weight = self.dual_frame_weight
        assert dual_weight is not None  # silence pyflakes
        inv_spects = []
        # if self._use_fftw:
        #     norm = self.__normalization
        #     # due to the way local variables are handled by ne.evaluate,
        #     # we have to use a loop
        #     for spect in self.spectrograms:
        #         inverse_spects.append(my_ifft_shift(
        # ne.evaluate('spect / dual_weight / norm')))
        # else:
        for spec in self.spectrograms:
            inv_spects.append(my_ifft_shift(ne.evaluate('spec / dual_weight')))
        return inv_spects

    def inverse_transform(self,
                          coeffs,
                          real=False,
                          inverse_spects=None,
                          do_norm=True):
        r"""
        Computes the inverse alpha-shearlet transform.

        Precisely, this method computes the **pseudo-inverse** of the
        alpha-shearlet transform. This is the same as first projecting onto
        the range of the alpha-shearlet transform and then inverting.

        In particular, we have the following::

            >>> my_trafo = AlphaShearletTransform(512, 512, [0.5]*3)
            >>> im = np.random.random((512, 512))
            >>> coeff = my_trafo.transform(im)
            >>> np.allclose(im, my_trafo.inverse_transform(coeff))
            True

        :param coeffs:
            The coefficients of which the inverse transform should be
            calculated.

            Just as for the method :func:`adjoint_transform`, ``coeffs``
            should be roughly of the same type as the return value
            of :func:`transform`. Usually, this argument will be obtained
            directly or indirectly from :func:`transform`, e.g. by
            thresholding. Fore more details, see the documentation of
            :func:`adjoint_transform`.

        :param bool real:
            Setting ``real=True`` will cause ``inverse_transform`` to only
            return the real part of the actual inverse transform. Hence, we
            always have::

                >>> np.allclose(my_trafo.inverse_transform(coeff),
                                np.real(my_trafo.inverse_transform(im,
                                                                   real=True)))
                True

            .. note:: This is *not* the same as passing ``real=True`` to the
                      constructor of the class :class:`AlphaShearletTransform`.

        :param bool do_norm:
            To obtain the (pseudo)-inverse to the method :func:`transform`,
            this parameter must be set to the same value as for calling
            :func:`transform`.

            More precisely, recall that passing ``do_norm=True`` to
            :func:`transform` causes the transform to be normalized, i.e., to
            compute :math:`f \ast \psi_i / \|\psi_i \|_{L^2}` instead of
            :math:`f \ast \psi_i`. If ``do_norm=True`` is passed to
            :func:`inverse_transform`, this normalization is undone and then
            the usual (pseudo)-inverse is applied.

        :param inverse_spects:
            This parameter contains the spectrograms used to compute the
            inverse transform. For the default value (``None``), the usual
            spectrograms of the canonical dual frame, i.e.,
            :math:`\tilde{\psi_i} = \mathcal{F}^{-1}[\widehat{\psi_i} / w]`
            are used, where ``w`` is the *dual frame weight*
            (cf. :func:`dual_frame_weight`).

            Essentially, this method simply computes
            :math:`\sum_i \mathrm{coeff}_i \ast \mathrm{inverse\_spects}_i`,
            which coincides with the pseudo inverse for the default choice of
            ``inverse_spects``.

            .. warning:: The parameter ``inverse_spects`` is mainly for
                         internal use or for performance optimizations.
                         Only use it if you really know what you are doing!
        """
        # pylint: disable=unused-variable
        # NOTE: We assume 'inverse_spects' to be my_ifft_shifted, see the
        #       definition of '_calculate_inverse_spectrograms'
        # since we are using ne.evaluate, so that the uses of some of the
        # variables are 'hidden' inside a string
        if do_norm:
            renormed_coeffs = (norm * c
                               for norm, c in zip(self.space_norms, coeffs))
        else:
            renormed_coeffs = coeffs

        if self.is_parseval:
            # result = self.adjoint_transform(coeffs, do_norm=do_norm)
            result = self.adjoint_transform(renormed_coeffs, do_norm=False)
        elif self.is_subsampled:
            result = self.adjoint_transform(renormed_coeffs,
                                            self.dual_spects,
                                            do_norm=False)
        else:

            # Do we really always want 'complex128'?
            result = np.zeros((self.height, self.width), dtype='complex128')
            if inverse_spects is None:
                dual_w = self.dual_frame_weight
                assert dual_w is not None  # silence pyflakes

                for i, (norm, coef, spec) in enumerate(zip(self.space_norms,
                                                           coeffs,
                                                           self.spectrograms)):
                    # coeff_fourier = my_fft_shift(fft2(fftshift(coef)))
                    coeff_f = my_fft_shift(self._fft(i, coef))
                    if do_norm:
                        ne.evaluate('result + spec * norm * coeff_f / dual_w',
                                    out=result)
                    else:
                        ne.evaluate('result + spec * coeff_f / dual_w',
                                    out=result)
                result = self._ifft(-1, my_ifft_shift(result))
            else:
                for norm, coeff, inv_spec in zip(self.space_norms,
                                                 coeffs,
                                                 inverse_spects):
                    # coeff_f = my_fft_shift(self._fft(-1, coeff))
                    coeff_f = self._fft(-1, coeff)
                    assert coeff_f is not None  # silence pyflakes
                    # result += ne.evaluate('inv_spec * coeff_f')
                    if do_norm:
                        ne.evaluate('result + inv_spec * coeff_f * norm',
                                    out=result)
                    else:
                        ne.evaluate('result + inv_spec * coeff_f', out=result)
                result = self._ifft(-1, result)
        if real:
            result = np.real(result)
        return result
