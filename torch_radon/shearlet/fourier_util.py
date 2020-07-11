#!/usr/bin/env python3
r"""
"""

import numpy.fft


def my_fft_shift(A):
    r"""
    An alternative fftshift implementation, which changes the orientation
    of the y-axis.

    Motivation: With the implementation of fft.fft2 and fft.fftshift of numpy,
    the y-axis is reversed - compared with the mathematical y-axis,
    which decreases from top to bottom.

    In contrast, the current implementation returns an array which obeys
    the mathematical orientation of the y-axis.

    Parameters:
        A : 2 dimensional np.array, probably obtained by numpy.fft.fft2.

    Returns:
        shifted version of A, such that the origin is in the "middle" of
        the screen (when displaying the array) and such that the y-axis
        decreases from top to bottom.
    """
    # this indeed only changes the order of the ROWS, not of the columns!
    return numpy.fft.fftshift(A)[::-1]


def my_ifft_shift(A):
    r"""
    Inverse of 'my_fft_shift'.
    """
    return numpy.fft.ifftshift(A[::-1])


def fft2(image):
    return numpy.fft.fft2(image, norm='ortho')


def ifft2(image):
    return numpy.fft.ifft2(image, norm='ortho')
