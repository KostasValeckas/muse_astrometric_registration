.. _installation:

Installation
~~~~~~~~~~~~~~

.. note::
    The software has been tested on Python 3.12.

The installation process consists of the following steps:

1. :ref:`Get the spacepylot submodule <submodule>`.
2. :ref:`Install the dependencies <dependencies>`.


.. _submodule:

Spacepylot submodule
--------------------

The repository relies on the ``spacepylot`` submodule, which is a
`fork of the original spacepylot repository <https://github.com/KostasValeckas/spacepylot/tree/5dfca0975f47b163739fc1d551baf6f2da0d2b15>`_.

If you have cloned the code using git, you need to initialize the submodule with:

.. code-block:: bash

    git submodule update --init --recursive

The submodule link is set up to use https, so if you want to use ssh instead,
you can change the link in the ``.gitmodules`` file.


.. _dependencies:

Installing dependencies
-----------------------

Extra dependencies are in the ``requirements.txt`` file. Install them with:

.. code-block:: bash

    pip install -r requirements.txt

TODO: a guide for creating a clean Python environment (conda / venv).
