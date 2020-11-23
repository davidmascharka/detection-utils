# detection-utils
> Written by [David Mascharka](https://github.com/davidmascharka), [Ryan Soklaski](https://github.com/rsokl), and [Arjun Majumdar](https://github.com/arjunmajum)

![build status](https://img.shields.io/travis/davidmascharka/detection-utils.svg) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4287380.svg)](https://doi.org/10.5281/zenodo.4287380)


Common functionality for object detection

This repository hosts functions that are commonly used across object detection projects. The
functionality here includes:

- [Box overlap computation](src/detection_utils/boxes.py#L29) (IoU)
- [Precision](src/detection_utils/metrics.py#L27) and [recall](src/detection_utils/metrics.py#L93) calculations
- [Computing targets](src/detection_utils/boxes.py#L82) for training a detector given a set of ground-truth objects
- [Non-maximum suppression](src/detection_utils/boxes.py#L170)
- [Coordinate transform utilities](src/detection_utils/boxes.py#L241)
- [Focal loss in PyTorch](src/detection_utils/pytorch.py#L26)

All the functions here are
[well-tested](tests) to ensure proper
functionality of these utilities. This repository is meant to ensure that modifications and improvements that are
implemented as part of one program migrate to other programs when appropriate.

## Installation

To use `detection-utils` you will need `numpy`, `numba`, and optionally `pytorch`. We recommend either fetching `numpy`
from [Anaconda](https://www.anaconda.com/distribution/) or ensuring you can link against MKL yourself for
performance. You can install `numba` via pip and [PyTorch](https://pytorch.org/get-started/locally/) should be installed
according to your needs. For example:

``` shell
$ conda install numpy
$ pip install numba torch torchvision
```

#### Pip

`detection-utils` is availabe on PyPI: install via

``` shell
$ pip install detection-utils
```

#### Git
Clone this repository and install:

``` shell
$ git clone https://github.com/davidmascharka/detection-utils.git && cd detection-utils
$ pip install .
```

#### Verifying Installation

To verify installation, ensure that [pytest](https://docs.pytest.org/en/latest/) and
[hypothesis](https://hypothesis.readthedocs.io/en/latest/) are installed, then run
`pytest` from the `detection-utils` base directory.

## Contributing
If you're interested in contributing to this library, a good place to start is by browsing for [open
issues](https://github.com/davidmascharka/detection-utils/issues) to see if there are
desired features waiting for implementation. Please see the [contributors file](CONTRIBUTING.md) before you submit a
pull request.

## Copyright
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research and
Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or
recommendations expressed in this material are those of the author(s) and do not necessarily
reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2019 Massachusetts Institute of Technology.

MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other
than as specifically authorized by the U.S. Government may violate any copyrights that exist in
this work.
