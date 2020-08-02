Radon Projections
==================

Parallel Beam
---------------


.. automodule:: torch_radon
   :noindex:


.. autoclass:: Radon

    .. automethod:: forward(self, x)

    .. automethod:: backprojection(self, sinogram)

    .. automethod:: backward(self, sinogram)


Fanbeam
---------------

.. autoclass:: RadonFanbeam

    .. automethod:: forward(self, x)

    .. automethod:: backprojection(self, sinogram)

    .. automethod:: backward(self, sinogram)
