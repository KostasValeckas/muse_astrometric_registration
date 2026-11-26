"""
Third-party code bundled with the package because it is not available on PyPI.

- ``spacepylot``: fork of spacepylot (https://github.com/KostasValeckas/spacepylot,
  MIT license, see ``spacepylot/LICENSE.txt``), the Python files of ``src/spacepylot``
  without the example data. Only its TV-L1 optical flow field and helpers are used
  (``methods.OpticalFlow``). Change it in the fork and copy it here again.

  Local changes (keep them when copying from the fork again):

  - ``utils.py``: ``scipy.odr`` (deprecated since SciPy 1.17, removed in 1.19) is
    imported inside ``regress_odr`` instead of at the top of the module, so
    importing spacepylot gives no deprecation warning. ``regress_odr`` (only used by
    the plotting helper ``_polynorm``) is never called by the pipeline.
"""
