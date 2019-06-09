.. detection-utils documentation master file, created by
   sphinx-quickstart on Fri Jun  7 12:24:38 2019.

detection-utils
===============
detection-utils provides utilities that are common across many object detection projects.
This includes things like:

- Box overlap computation (IoU)
- Precision and recall calculations
- Computing targets for training a detector given a set of ground-truth objects and anchor boxes
- Non-maximum suppression
- Coordinate transformation utilities
- Focal loss

All the functions here are well-tested to ensure proper functionality and are used in real projects, so are intended to be enterprise-grade. This repository is meant to ensure that modifications and improvements that are implemented in one object detection project migrate to other projects as appropriate, to reduce code duplication, and to enable a quicker start to working on object detection.

An `example project <https://github.com/davidmascharka/detection-utils/blob/master/examples/example-detector.ipynb>`_ is included that walks through using detection-utils for a simple object detection problem from scratch. While not intended as a full introduction to the problem of object detection, there is enough exposition for an accelerated introduction to the problem.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   documentation
   changes


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
