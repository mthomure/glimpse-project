glimpse.backends
================

.. module:: glimpse.backends

.. data:: ACTIVATION_DTYPE

   Element type for an array of Glimpse activation values.

.. data:: DEFAULT_BACKEND_NAME

   Backend name used by MakeBackend when no name is supplied.

.. autofunction:: MakeBackend

.. autoclass:: InputLoadError

.. autoclass:: InputSource

.. autoexception:: BackendError

.. autoexception:: InputSizeError

.. autoclass:: IBackend
   :members:

:mod:`base_backend` Module
--------------------------

.. module:: glimpse.backends.base_backend

.. autoclass:: BaseBackend
   :no-members:

:mod:`scipy_backend` Module
---------------------------

.. module:: glimpse.backends.scipy_backend

.. autoclass:: ScipyBackend
   :no-members:
