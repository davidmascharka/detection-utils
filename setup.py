from setuptools import setup, find_packages
import versioneer

DISTNAME = 'detection-utils'
DESCRIPTION = 'Common functionality for object detection'
LICENSE = 'MIT'
AUTHOR = 'David Mascharka, Ryan Soklaski, Arjun Majumdar'
AUTHOR_EMAIL = 'davidmascharka@gmail.com, rsoklaski@gmail.com, arjun.majum@gmail.com'
URL = 'https://github.com/fill/this/in/later/detection-utils'
INSTALL_REQUIRES = ['numpy >= 1.13', 'torch >= 0.4', 'numba >= 0.38']
TESTS_REQUIRE = ['pytest >= 3.8', 'hypothesis >= 4.6']

LONG_DESCRIPTION = """
detection-utils provides utilities that are common across many object detection projects.
This includes things like:

- Box overlap computation (IoU)
- Precision and recall calculations
- Computing targets for training a detector given a set of ground-truth objects and anchor boxes
- Non-maximum suppression
- Coordinate transformation utilities
- Focal loss

All the functions here are well-tested to ensure proper functionality and are used in real projects, so are
intended to be enterprise-grade. This repository is meant to ensure that modifications and improvements that
are implemented in one object detection project migrate to other projects as appropriate, to reduce code
 duplication, and to enable a quicker start to working on object detection.
"""

if __name__ == '__main__':
    setup(name=DISTNAME,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          license=LICENSE,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          instsall_requires=INSTALL_REQUIRES,
          tests_require=TESTS_REQUIRE,
          url=URL,
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          python_requires='>=3.6',
          packages=find_packages(where='src', exclude=['tests*']),
          package_dir={'': 'src'},
          )
