.. _developer:

Interested in developing?
==========================

TODO: how to contribute.

A few things are pre-determined:

#. We use `GitHub <https://github.com/KostasValeckas/muse_astrometric_registration>`_ for version control.
#. Tests are in the ``tests`` directory.
#. Development scripts and tools that are not part of the pipeline are in the ``dev`` directory.

Building the documentation
--------------------------

The documentation is built with `Sphinx <https://www.sphinx-doc.org/>`_:

.. code-block:: bash

    pip install sphinx
    cd docs
    make html

The pages are then in ``docs/_build/html``.
