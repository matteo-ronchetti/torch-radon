#!/usr/bin/env python3
r"""
The classes, functions and objects in this module make it (comparatively)
easy to define custom generating functions which act as "Mother shearlets".

.. warning::
    Implementing custom "Mother shearlets" is a complex task and thus should
    only be done if really needed. Proceed with caution.

To implement a custom set of generating functions, all one needs to do is
to create a suitable object of the class :class:`MotherShearlet` and pass
it to the constructor of the :class:`AlphaTransform.AlphaShearletTransform`,
via the ``mother_shearlet`` parameter.

In a nutshell, in our implementation, a shearlet system is determined by

* A **low-pass filter** which is given by :math:`\phi \otimes \phi`, where
  :math:`\phi` is given by the ``low_pass_function`` attribute of class
  :class:`MotherShearlet`.

* A **mother shearlet** :math:`\psi`, which in our case is (in Fourier domain)
  given by

  .. math::
      \widehat{\psi}(\xi) = \psi_1 (\xi_1) \cdot \psi_2 (\xi_2 / \xi_1).

  Here,

  * we call :math:`\psi_1` the **scale-sensitive generating function**.
    It is given by the ``scale_function``` attribute of class
    :class:`MotherShearlet`.

  * we call :math:`\psi_2` the **direction sensitive generating function**.
    It is given by the ``direction_function`` attribute of class
    :class:`MotherShearlet`.

See the documentation of :class:`MotherShearlet` below for more details.
"""

from collections import namedtuple
import numpy as np
import numexpr as ne

# -----------------------------------------------------------------------------
#               Definition of BumpFunction-related classes
# -----------------------------------------------------------------------------

BumpFunction = namedtuple('BumpFunction', ["support",
                                           "large_support",
                                           "call",
                                           "theano_call"])

BumpFunction.__doc__ = (
    r"""
    Each instance of this class encapsulates a specific (one-dimensional)
    bump function ``f``.

    **Attributes**:

    .. py:attribute:: support

       A tuple ``(a,b)`` such that the bump function ``f``
       modeled by this object satisfies
       :math:`\mathrm{supp} f \subset (a,b)`.

    .. py:attribute:: large_support

       A tuple ``(a,b)`` such that the bump function ``f`` modeled
       by this object is "large" on :math:`(a,b)`, i.e.,
       :math:`|f| \geq c > 0` on :math:`(a,b)` for a (reasonably large)
       constant :math:`c>0`, e.g. :math:`c = \frac{1}{2}`.

    .. py:attribute:: call

       A callable object calculating ``f(x)`` given ``x``.
       This should be vectorized, i.e., ``call()`` should accept
       numpy arrays (:class:`numpy.ndarray`) and apply ``f``
       componentwise.

    .. py:attribute:: theano_call

       This should be a callable object doing the same as ``call``,
       but working on :class:`theano.tensor.dmatrix` objects.

       .. note::
          Currently, this attribute is not used anywhere and can thus
          be set arbitrarily (e.g. to ``None``).
""")

MotherShearlet = namedtuple('MotherShearlet', ["low_pass_function",
                                               "scale_function",
                                               "direction_function"])
MotherShearlet.__doc__ = (
    r"""\
    Each instance of this class acts as a container for the three (main)
    components determining an alpha-shearlet system.

    The actual *mother shearlet* :math:`\psi` is given by

    .. math::
        \psi(\xi) = \psi_1 (\xi_1) \cdot \psi_2 (\xi_2 / \xi_1),

    where:
        * :math:`\psi_1` is given by the ``scale_function`` attribute,
        * :math:`\psi_2` is given by the ``direction_function`` attribute.

    Finally, the *low frequency component* of the shearlet system is given by
    :math:`\phi \otimes \phi` (tensor product),
    where :math:`\phi` is given by the ``low_pass_function`` attribute.

    .. note::
        All attributes are expected to be of type :class:`BumpFunction`.

        Finally,

        1. if ``(a,b) = scale_function.large_support``, then it is expected
           that:

           i.  ``a,b > 0``
           ii. ``b/a >= 2``.

        2. if ``(a,b) = direction_function.large_support``, it is expected
           that:

           i.  ``b >= 1/2``,
           ii. ``a <= -1/2``.
    """)


# -----------------------------------------------------------------------------
#             Definition of BumpFunction-related helper functions
# -----------------------------------------------------------------------------


def translate(f, x_0):
    """
    This function translates a :class:`BumpFunction` object ``f``, i.e.,
    it returns :math:`L_{x_0} f`, where :math:`(L_y f)(x) = f(x - y)`.
    The new support is thus :math:`x_0 + \mathrm{supp} f`.

    *Parameters:*

    :param BumpFunction f:
        The :class:`BumpFunction` object which should be translated.

    :param float x_0:
        The translation parameter.

    *Return value:*

    :return:
        A new :class:`BumpFunction` object representing the translated
        function.
    """
    support = f.support
    newSupport = (support[0] + x_0, support[1] + x_0)
    large_support = f.large_support
    newLargeSupport = (large_support[0] + x_0, large_support[1] + x_0)
    return BumpFunction(newSupport,
                        newLargeSupport,
                        # lambda x: f.call(x - x_0),
                        lambda x, y=x_0: f.call(ne.evaluate('x - y')),
                        lambda x: f.theano_call(x - x_0))


def scale(f, a):
    """
    This function scales a :class:`BumpFunction` object ``f``,
    i.e., it returns :math:`f \circ (x \mapsto x/a)`.
    The new support is thus :math:`a \cdot \mathrm{supp} f`.

    *Parameters:*

    :param BumpFunction f:
        The :class:`BumpFunction` object which should be translated.

    :param float a:
        The scale parameter (positive real number).

    *Return value:*

    :return:
        A new :class:`BumpFunction` object representing the scaled function.
    """
    assert a > 0, "Scale parameter must be positive."
    support = f.support
    newSupport = (a * support[0], a * support[1])
    large_support = f.large_support
    newLargeSupport = (a * large_support[0], a * large_support[1])
    return BumpFunction(newSupport,
                        newLargeSupport,
                        # lambda x: f.call(x / a),
                        lambda x, y=a: f.call(ne.evaluate('x / y')),
                        lambda x: f.theano_call(x / a))


def flip(f):
    """
    This function "flips" a :class:`BumpFunction` object ``f``,
    i.e., it returns :math:`f \circ (x \mapsto -x)`.
    The new support is thus :math:`- \mathrm{supp} f`.

    *Parameters:*

    :param BumpFunction f:
        The :class:`BumpFunction` object which should be flipped.

    *Return value:*

    :return:
        A new 'BumpFunction' object representing the flipped function.
    """
    support = f.support
    newSupport = (-support[1], -support[0])
    large_support = f.large_support
    newLargeSupport = (-large_support[1], -large_support[0])
    return BumpFunction(newSupport,
                        newLargeSupport,
                        lambda x: f.call(-x),
                        lambda x: f.theano_call(-x))


# -----------------------------------------------------------------------------
#             Definition of Meyer mother shearlet
# -----------------------------------------------------------------------------


def meyer(A):
    r"""
    This function implements the function ``v`` from the paper |FFST|.

    It rises smoothly between 0 and 1.
    In particular, :math:`v(x) = 0` for :math:`x \leq 0` and
    :math:`v(x) = 1` for :math:`x \geq 1`.

    *Parameters:*

    :param numpy.ndarray A:
        The input (matrix with real entries).

    *Return value:*

    :return:
        :math:`v(A)`, i.e. the function ``v`` applied to each entry of ``A``.
    """
    # first, compute various powers of A
    # A2 = np.square(A)
    # A4 = np.square(A2)
    # A5 = A4 * A
    # A6 = A2 * A4
    # A7 = A6 * A

    # # AlmostResult is the result of Meyer's function for x with 0 < x < 1.
    # AlmostResult = 35 * A4 - 84 * A5 + 70 * A6 - 20 * A7

    # # Set result to 0 for values <= 0 and results to 1 for values >= 1.
    # PositiveMask = np.greater(A, 0)
    # LessThan1Mask = np.less(A, 1)

    # Result = AlmostResult * PositiveMask * LessThan1Mask + (1-LessThan1Mask)
    # return Result
    return ne.evaluate('((0 <= A) & (A < 1))'
                       '* (35 * A**4 - 84 * A**5 + 70 * A**6 - 20 * A**7)'
                       '+ (A >= 1)')


def meyer_low_pass(A):
    r"""
    This function uses the :func:`meyer` function to construct a bump function
    :math:`\phi` with the following properties:

        #. :math:`\mathrm{supp} \phi \subset [-2,2]`,
        #. :math:`\phi(x) = 1` for :math:`x \in (-3/2, 3/2)`.

    *Parameters:*

    :param numpy.ndarray A:
        The input (matrix with real entries).

    *Return value:*

    :return:
        :math:`\phi(A)`, i.e. the function :math:`\phi` applied to each entry
        of ``A``. Precisely, :math:`\phi(x) = 4 - 2 * v(|x|)`, where ``v`` is
        the function :func:`meyer`.
    """
    return meyer(4 - 2 * np.abs(A))


def meyer_scale_function(R):
    r"""
    This is an implementation of the function :math:`\tilde{W}` from the paper
    |Cartoon|.

    The main properties of this function are the following:
        #. :math:`\mathrm{supp} \tilde{W} \subset [1/2, 2]`.
        #. :math:`\tilde{W} \equiv 1` on :math:`[3/4, 3/2]`.

    .. note::
        The quotient of the bounds of the interval where :math:`\tilde{W}`
        is 1 is precisely 3/2 * 4/3 = 2, so that the function is suitable as
        a scale_function for the :class:`MotherShearlet` class.

    *Parameters:*

    :param numpy.ndarray R:
        The input (matrix with real entries).

    *Return value:*

    :return:
        :math:`\tilde{W}(R)`, i.e., :math:`\tilde{W}` applied componentwise.
    """
    # The "actual" implementation (implemented below) has support in [0, 1.5].
    # We shift the input by 0.5 to obtain the correct placement.
    R = R - 0.5
    RTimes4 = 4 * R

    PositiveMask = np.greater(R, 0)
    LessThan1FourthMask = np.less_equal(RTimes4, 1)
    LessThan5FourthMask = np.less_equal(RTimes4, 5)
    GreaterThan1Mask = np.greater(R, 1)

    # 'Rise' corresponds to meyer(4*x) if 0 < x <= 1/4
    Rise = meyer(RTimes4) * PositiveMask * LessThan1FourthMask
    # 'BumpValue' corresponds to the value \tilde{W}(x) = 1 if 1/4 < x <= 1
    BumpValue = (1 - LessThan1FourthMask) * (1 - GreaterThan1Mask)
    # 'Fall' corresponds to meyer(5 - 4*x) if 1 < x <= 5/4
    Fall = meyer(5 - RTimes4) * GreaterThan1Mask * LessThan5FourthMask

    return Rise + BumpValue + Fall


def meyer_direction_function(A):
    r"""
    This is an implementation of a direction function ``f``, i.e., it can be
    used for the attribute ``direction_function`` of the class
    :class:`MotherShearlet`.

    The main properties of this function are the following:
        #. :math:`\mathrm{supp} f \subset (-1,1)`,
        #. :math:`f \equiv 1` on :math:`(-1/2, 1/2)`,
        #. ``f`` is symmetric.

    The implementation uses the :func:`meyer` function.

    *Parameters:*

    :param numpy.ndarray A:
        The input (matrix with real entries).

    *Return value:*

    :return:
        ``f(A)``, i.e. ``f`` is applied componentwise.
    """
    negative_part = meyer(1 + A)
    positive_part = meyer(1 - A)

    positive_mask = np.greater(A, 0)
    negative_mask = 1 - positive_mask

    return positive_mask * positive_part + negative_mask * negative_part


# we set the 'theano_call' component to None to decouple from the theano
# dependency. If theano functionality is desired, import the module
# 'MotherShearletsTheano', which will change the 'theano_call'
# components from 'None' to the actual implementation.
_MeyerScaleFunction = BumpFunction((1 / 2, 2),  # support
                                   (3 / 4, 3 / 2),  # large_support
                                   meyer_scale_function,
                                   None)

_MeyerDirectionFunction = BumpFunction((-1, 1),  # support
                                       (-1 / 2, 1 / 2),  # large_support
                                       meyer_direction_function,
                                       None)

_MeyerLowPass = BumpFunction((-2, 2),  # support
                             (-3 / 2, 3 / 2),  # large_support
                             meyer_low_pass,
                             None)

MeyerMotherShearlet = MotherShearlet(_MeyerLowPass,
                                     _MeyerScaleFunction,
                                     _MeyerDirectionFunction)


# -----------------------------------------------------------------------------
#             Definition of Haeuser mother shearlet
# -----------------------------------------------------------------------------


def haeuser_scale_function(A):
    r"""
    This is an implementation of the function ``b`` from the paper |FFST|,
    but with support restricted to *positive* numbers, whereas in the paper,
    the function is symmetric.

    This function is supported in [1,4], is "large" on [2,3] and satisfies
    :math:`\sum_{j=-\infty}^\infty b(2^j x) = 1` for all :math:`x > 0`.

    Furthermore, :math:`b(x) \geq \frac{1}{\sqrt{2}} \geq \frac{1}{2}` for
    :math:`x \in [1.5, 3]`.  Thus, ``b`` is a good candidate for
    a ``scale_function``, cf. :class:`MotherShearlet`.

    *Parameters:*

    :param numpy.ndarray A:
        The input (a matrix with real entries).

    *Return value:*

    :return:
        ``b(A)``, i.e., ``b`` is applied componentwise.
    """
    # GreaterThan1Mask = np.greater(A, 1)
    # GreaterThan2Mask = np.greater(A, 2)
    # LessThan2Mask = 1 - GreaterThan2Mask
    # LessThan4Mask = np.less_equal(A, 4)

    # firstBranch = np.sin((np.pi / 2) * meyer(A - 1))
    # secondBranch = np.cos((np.pi / 2) * meyer(A/2 - 1))

    # return (GreaterThan1Mask * LessThan2Mask * firstBranch +
    #         GreaterThan2Mask * LessThan4Mask * secondBranch)
    pi = np.pi
    B = meyer(A - 1)
    C = meyer(A / 2 - 1)
    return ne.evaluate('((A > 1) & (A <= 2)) * sin(pi/2 * B)'
                       '+ ((A > 2) & (A <= 4)) * cos(pi/2 * C)')


def haeuser_direction_function(A):
    r"""
    This is an implementation of the function :math:`\widehat{\psi_2}`
    from the paper |FFST| (cf. eq. (6) in that paper).

    This function has support in [-1,1], is "large" on [-1/2,1/2] and satisfies
    :math:`\sum_{j=-\infty}^\infty |\widehat{\psi_2}(x + j)|^2 = 1` for all x.

    Thus, :math:`\widehat{\psi_2}` is a good candidate for a
    ``direction_function``, cf. :class:`MotherShearlet`.

    *Parameters:*

    :param numpy.ndarray A:
        The input (a matrix with real entries).

    *Return value*

    :return:
        :math:`\widehat{\psi_2}(A)`, i.e., :math:`\widehat{\psi_2}` is
        applied componentwise.
    """
    # first_branch = np.sqrt(meyer(1 + A))
    # second_branch = np.sqrt(meyer(1 - A))

    # positive_mask = np.greater(A, 0)
    # negative_mask = 1 - positive_mask

    # return negative_mask * first_branch + positive_mask * second_branch
    B = meyer(1 + A)
    C = meyer(1 - A)
    return ne.evaluate('(A <= 0) * sqrt(B) + (A > 0) * sqrt(C)')


_HaeuserScaleFunction = BumpFunction((1, 4),  # support
                                     (3 / 2, 3),  # large_support
                                     haeuser_scale_function,
                                     None)

_HaeuserDirectionFunction = BumpFunction((-1, 1),  # support
                                         (-1 / 2, 1 / 2),  # large_support
                                         haeuser_direction_function,
                                         None)

# we use the meyer low pass filter also for the haeuser mother shearlet.
HaeuserMotherShearlet = MotherShearlet(_MeyerLowPass,
                                       _HaeuserScaleFunction,
                                       _HaeuserDirectionFunction)


# -----------------------------------------------------------------------------
#             Definition of indicator mother shearlet
# -----------------------------------------------------------------------------


def indicator_scale_function(A):
    r"""
    Vectorized implementation of the indicator function :math:`\chi_{[1,2]}`,
    which can be used as a ``scale_function``, mainly for testing purposes
    (since the space localization is horrible).

    *Parameters:*

    :param numpy.ndarray A:
        The input (matrix with real entries).

    *Return value*:

    :return:
        :math:`\chi_{[1,2]}(A)`, applied componentwise.
    """
    greater_than_1_mask = np.greater_equal(A, 1)
    less_than_2_mask = np.less_equal(A, 2)
    return greater_than_1_mask * less_than_2_mask


def indicator_low_pass_function(A):
    r"""
    Vectorized implementation of the indicator function :math:`\chi_{[-1,1]}`,
    which can be used as a ``low_pass_function``, mainly for testing purposes
    (since the space localization is horrible).

    *Parameters:*

    :param numpy.ndarray A:
        The input (matrix with real entries).

    *Return value*:

    :return:
        :math:`\chi_{[-1,1]}(A)`, applied componentwise.
    """
    less_than_1_mask = np.less_equal(A, 1)
    greater_than_minus1_mask = np.greater_equal(A, -1)
    return less_than_1_mask * greater_than_minus1_mask


_IndicatorScaleFunction = BumpFunction((1, 2),  # support
                                       (1, 2),  # large_support
                                       indicator_scale_function,
                                       None)

_IndicatorLowPass = BumpFunction((-1, 1),  # support
                                 (-1, 1),  # large_support
                                 indicator_low_pass_function,
                                 None)

_IndicatorDirectionFunction = scale(_IndicatorLowPass, 0.5)

IndicatorMotherShearlet = MotherShearlet(_IndicatorLowPass,
                                         _IndicatorScaleFunction,
                                         _IndicatorDirectionFunction)
