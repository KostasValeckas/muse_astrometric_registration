.. _installation:

Installation
~~~~~~~~~~~~~~

.. note::
    The software has been tested on Python 3.12.

The software is run by executing Python scripts from the repository in a
terminal. The installation process consists of the following steps:

1. :ref:`Create a clean Python environment <venv>` (**optional - but strongly
   recommended**).
2. :ref:`Get the code and the spacepylot submodule <submodule>`.
3. :ref:`Install the dependencies <dependencies>`.

We provide both a quick guide for experienced Python users and a more detailed
guide for users who are new to Python environments.

Quick guide for experienced Python users
-----------------------------------------

1. Create and activate a new Python 3.12 environment.
2. Clone the repository with its submodule and install the dependencies from the
   repository root:

.. code-block:: bash

    git clone --recurse-submodules https://github.com/KostasValeckas/muse_astrometric_registration.git
    cd muse_astrometric_registration
    pip install -r requirements.txt


Detailed guide for users new to Python environments
----------------------------------------------------

.. _venv:

1. Create a clean Python environment
====================================

To ensure the best possible stability of the software and to avoid version
conflicts with other Python packages on your system, it is **strongly
recommended** to create a clean Python environment for running the software.
You can skip directly to
:ref:`getting the code <submodule>` if you prefer not to use a clean environment -
in that case you might experience software bugs due to version conflicts that are
not accounted for in this documentation.

**Using Anaconda (conda) (recommended):**

To create a new virtual environment using Anaconda, run the following command in
your terminal (if you are using Windows, do this and all following commands from
the Anaconda Prompt):

.. code-block:: bash

    conda create --name muse_registration python=3.12

You can replace ``muse_registration`` with any name you like. This will create a
new environment with Python 3.12 installed.

To activate the environment, run:

.. code-block:: bash

    conda activate muse_registration

**Using venv (standard Python):**

To create a new virtual environment using venv (standard Python), make sure you
have Python 3.12 installed, then run the following command in your terminal:

.. code-block:: bash

    python3.12 -m venv muse_registration

You can replace ``muse_registration`` with any name you like. This will create a
new environment with Python 3.12 installed.

.. note::

    If you are using Windows, you might need to run the following command instead:

    .. code-block:: powershell

        python -m venv muse_registration

    This is because the Python executable might not be named ``python3.12`` on
    Windows. In that case, you can check which version of Python is used by
    running:

    .. code-block:: powershell

        python --version

    If the Python version printed is not 3.12, you have several options:

    1. If your version is close to 3.12, you will most likely be fine. Otherwise,
       try one of the following steps.
    2. Install Anaconda and create the environment using the conda command as
       described above.
    3. Set the Python version used by the terminal by adding the Python
       installation directory to the PATH environment variable. See
       `How to add Python to PATH <https://realpython.com/add-python-to-path/>`_
       for more information.

To activate the environment, run:

For Linux/MacOS:

.. code-block:: bash

    source muse_registration/bin/activate

where ``muse_registration/bin/activate`` is the path to the activate script in the
environment.

For Windows:

.. code-block:: powershell

    # In PowerShell
    .\muse_registration\Scripts\Activate.ps1

    # In cmd.exe
    .\muse_registration\Scripts\activate.bat

where ``muse_registration\Scripts`` is the path to the activate scripts in the
environment.

.. _submodule:

2. Get the code and the spacepylot submodule
============================================

The repository relies on the ``spacepylot`` submodule, which is a
`fork of the original spacepylot repository <https://github.com/KostasValeckas/spacepylot/tree/5dfca0975f47b163739fc1d551baf6f2da0d2b15>`_.
You don't need to download it separately: git fetches it together with the
pipeline code.

Clone the repository together with the submodule:

.. code-block:: bash

    git clone --recurse-submodules https://github.com/KostasValeckas/muse_astrometric_registration.git

If you have already cloned the repository without the submodule, initialize it
from the repository root with:

.. code-block:: bash

    git submodule update --init --recursive

The submodule link is set up to use https, so if you want to use ssh instead,
you can change the link in the ``.gitmodules`` file.


.. _dependencies:

3. Install the dependencies
===========================

With your environment activated, go to the repository root (the directory with
the file ``requirements.txt`` in it) and install the dependencies, including the
``spacepylot`` submodule:

.. code-block:: bash

    cd muse_astrometric_registration
    pip install -r requirements.txt

The submodule has to be initialized first (step 2): ``requirements.txt`` installs
it from the local ``./spacepylot`` directory.

After the installation is complete, you can check that everything is in place by
running:

.. code-block:: bash

    python src/run.py --help

If the installation worked, this prints the usage of the
:ref:`single method run <run>`. Continue with :ref:`getting started
<getting_started>`.
