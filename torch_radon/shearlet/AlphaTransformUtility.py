r"""
This module contains several utility functions which can be used e.g.
for thresholding the alpha-shearlet coefficients or for using the
alpha-shearlet transform for denoising.

Finally, it also contains the functions :func:`my_ravel` and :func:`my_unravel`
which can be used to convert the alpha-shearlet coefficients into a
1-dimensional vector and back. This is in particular convenient for the
subsampled transform, where this conversion is not entirely trivial, since the
different "coefficient images" have varying dimensions.
"""


import os.path
import math
import numpy as np
import numexpr as ne
import scipy.ndimage


def find_free_file(file_template):
    r"""
    This function finds the first nonexistent ("free") file obtained by
    "counting upwards" using the passed template/pattern.

    **Required Parameter**

    :param string file_template:
        This should be a string whose ``format()`` method can be called
        using only an integer argument, e.g. ``'/home/test_{0:0>2d}.txt'``,
        which would result in ``find_free_file`` consecutively checking
        the following files for existence:

        `/home/test_00.txt,`
        `/home/test_01.txt, ...`

    **Return value**

    :return:
        ``file_template.format(i)`` for the first value of ``i`` for which
        the corresponding file does not yet exist.
    """
    i = 0
    while os.path.isfile(file_template.format(i)):
        i += 1
    return file_template.format(i)


def threshold(coeffs, thresh_value, mode):
    r"""
    Given a set of coefficients, this function performs a thresholding
    procedure, i.e., either soft or hard thresholding.

    **Required parameters**

    :param coeffs:
        The coefficients to be thresholded.

        Either a three-dimensional :class:`numpy.ndarray` or a generator
        producing two dimensional :class:`numpy.ndarray` objects.

    :param float thresh_value:
        The thresholding cutoff :math:`c` for the coefficients, see also
        ``mode`` for more details.

    :param string mode:
        Either ``'hard'`` or ``'soft'``. This parameter determines whether
        the hard thresholding operator

        .. math::
            \Lambda_cx
            =\begin{cases}
                x, & \text{if }|x|\geq c,\\
                0, & \text{if }|x|<c,
             \end{cases}

        or the soft thresholding operator

        .. math::
            \Lambda_cx
            =\begin{cases}
                x\cdot \frac{|x|-c}{|x|}, & \text{if }|x|\geq c,\\
                0,                        & \text{if }|x|<c
             \end{cases}

        is applied to each entry of the coefficients.

    **Return value**

    :return:
        A generator producing the thresholded coefficients. Each
        thresholded "coefficient image", i.e., each thresholded
        2-dimensional array, is produced in turn.
    """
    if mode == 'hard':
        for coeff in coeffs:
            ev_string = 'coeff * (real(abs(coeff)) >= thresh_value)'
            yield ne.evaluate(ev_string)
            # yield coeff * (np.abs(coeff) >= thresh_value)
    elif mode == 'soft':
        for coeff in coeffs:
            ev_string = ('(real(abs(coeff)) - thresh_value) * '
                         '(real(abs(coeff)) >= thresh_value)')
            large_values = ne.evaluate(ev_string)
            # large_values = np.maximum(np.abs(coeff) - thresh_value, 0)
            ev_str_2 = 'coeff * large_values / (large_values + thresh_value)'
            yield ne.evaluate(ev_str_2)
            # yield coeff * large_values / (large_values + thresh_value)
    else:
        raise ValueError("'mode' must be 'hard' or 'soft'")


def scale_gen(trafo):
    r"""
    **Required parameter**

    :param trafo:
        An object of class :class:`AlphaTransform.AlphaShearletTransform`.

    **Return value**

    :return:
        A generator producing integers. The i-th produced integer
        is the *scale* (starting from -1 for the low-pass part) of the i-th
        alpha-shearlet associated to ``trafo``.

        Hence, if ``coeff = trafo.transform(im)``, then the following iteration
        produces the associated scale to each "coefficient image"::

            for scale, c in zip(scale_gen(trafo), coeff):
                ...

    """
    indices_gen = iter(trafo.indices)
    next(indices_gen)
    yield -1
    for index in indices_gen:
        yield index[0]


def denoise(img, trafo, noise_lvl, multipliers=None):
    r"""
    Given a noisy image :math:`\tilde f`, this function performs a denoising
    procedure based on shearlet thresholding. More precisely:

    #. A scale dependent threshold parameter :math:`c=(c_j)_j` is calculated
       according to :math:`c_j=m_j\cdot \lambda / \sqrt{N_1\cdot N_2}`, where
       :math:`m_j` is a  multiplier for the jth scale, :math:`\lambda` is the
       noise level present in the image :math:`\tilde f` and
       :math:`N_1\times N_2` are its dimensions.

    #. The alpha-shearlet transform of :math:`\tilde f` is calculated
       using ``trafo``.

    #. Hard thesholding with threshold parameter (cutoff) :math:`c` is
       performed on alpha-shearlet coefficients, i.e., for each scale ``j``,
       each of the coefficients belonging to the jth scale is set to zero if
       its absolute value is smaller than :math:`c_j` and otherwise it is
       left unchanged.

    #. The (pseudo)-inverse of the alpha-shearlet transform is applied to the
       thresholded coefficients and this reconstruction is the return value
       of the function.

    **Required parameters**

    :param numpy.ndarray img:
        The “image” (2 dimensional array) that should be denoised.

    :param trafo:
        An object of class :class:`AlphaTransform.AlphaShearletTransform`.
        This object is used to calculate the (inverse) alpha-shearlet
        transform during the denoising procedure.

        The dimension of the transform and of ``img`` need to coincide.

    :param float noise_lvl:
        The (presumed) noise level present in ``img``.
        If ``img = img_clean + noise``, then ``noise_lvl`` should be
        approximately equal to the :math:`\ell^2` norm of ``noise``.

        In particular, if ``im`` is obtained by adding Gaussian noise with
        standard deviation :math:`\sigma` (in each entry) to a noise free
        image :math:`f`, then the noise level :math:`\lambda` is given by
        :math:`\lambda= \sigma\cdot \sqrt{N_1\cdot N_2}`; see also
        :func:`AdaptiveAlpha.optimize_denoising`.

    **Keyword parameter**

    :param list multipliers:
        A list of multipliers (floats) for each scale. ``multipliers[j]``
        determines the value of :math:`m_j` and thus of the cutoff
        :math:`c_j = m_j \cdot \lambda / \sqrt{N_1 \cdot N_2}` for scale ``j``.

        In particular, ``len(multipliers)`` needs
        to be equal to the number of the  scales of ``trafo``.

    **Return value**

    :return:
        The denoised image, i.e., the result of the denoising procedure
        described above.
    """
    coeff_gen = trafo.transform_generator(img, do_norm=True)

    if multipliers is None:
        # multipliers = [1] + ([2.5] * (trafo.num_scales - 1)) + [5]
        multipliers = [3] * trafo.num_scales + [4]

    width = trafo.width
    height = trafo.height
    thresh_lvls = [multi * noise_lvl / math.sqrt(width * height)
                   for multi in multipliers]

    thresh_coeff = (coeff * (np.abs(coeff) >= thresh_lvls[scale + 1])
                    for (coeff, scale) in zip(coeff_gen, scale_gen(trafo)))
    recon = trafo.inverse_transform(thresh_coeff, real=True, do_norm=True)
    return recon


def image_load(path):
    r"""
    Given a  '.npy' or '.png' file, this function loads the file and returns
    its content as a two-dimensional :class:`numpy.ndarray` of :class:`float`
    values.

    For '.png' images, the pixel values are normalized to be between 0 and 1
    (instead of between 0 and 255) and color images are converted to
    grey-scale.

    **Required parameter**

    :param string path:
        Path to the image to be converted, either of a '.png' or '.npy' file.

    **Return value**

    :return:
        The loaded image as a two-dimensional :class:`numpy.ndarray`.

    """
    image_extension = path[path.rfind('.'):]

    if image_extension == '.npy':
        return np.array(np.load(path), dtype='float64')
    elif image_extension == '.png':
        return np.array(scipy.ndimage.imread(path, flatten=True) / 255.0,
                        dtype='float64')
    else:
        raise ValueError("This function can only load .png or .npy files.")


def _print_listlist(listlist):
    for front, back, l in zip(['['] + ([' '] * (len(listlist) - 1)),
                              ([''] * (len(listlist) - 1)) + [']'],
                              listlist):
        print(front + str(l) + back)


def my_ravel(coeff):
    r"""
    The subsampled alpha-shearlet transform returns a list of differently
    sized(!) two-dimensional arrays. Likewise, the fully sampled transform
    yields a three dimensional numpy array containing the coefficients.
    The present function can be used (in both cases) to convert this list into
    a single *one-dimensional* numpy array.

    .. note::
        In order to invert this conversion to a one-dimensional array,
        use the associated function :func:`my_unravel`. Precisely,
        :func:`my_unravel` satisfies
        ``my_unravel(my_trafo, my_ravel(coeff)) == coeff``,
        if coeff is obtained from calling ``my_trafo.transform(im)``
        for some image ``im``.

        The preceding equality holds at least up to (negligible)
        differences (the left-hand side is a generator while the
        right-hand side could also be a list).

    **Required parameter**

    :param list coeff:
        A list (or a generator) containing/producing two-dimensional
        numpy arrays.

    **Return value**

    :return:
        A one-dimensional :class:`numpy.ndarray` from which **coeff** can
        be reconstructed.
    """
    return np.concatenate([c.ravel() for c in coeff])


def my_unravel(trafo, coeff):
    r"""
    This method is a companion method to :func:`my_ravel`.
    See the documentation of that function for more details.

    **Required parameters**

    :param trafo:
        An object of class :class:`AlphaTransform.AlphaShearletTransform`.

    :param numpy.ndarray coeff:
        A one-dimensional numpy array, obtained via
        ``my_ravel(coeff_unrav)``, where ``coeff_unrav`` is of the same
        dimensions as the output of ``trafo.transform(im)``, where
        ``im`` is an image.

    **Return value**

    :return:
        A generator producing the same values as ``coeff_unrav``, i.e.,
        an "unravelled" version of ``coeff``.
    """
    coeff_sizes = [spec.shape for spec in trafo.spectrograms]
    split_points = np.cumsum([spec.size for spec in trafo.spectrograms])
    return (c.reshape(size)
            for size, c in zip(coeff_sizes, np.split(coeff, split_points)))
