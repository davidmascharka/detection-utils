# detection-utils
Common functionality for object detection

This repository hosts functions that are commonly used across object detection projects. The
functionality here includes:

- [Box overlap computation](src/detection_utils/boxes.py#L6) (IoU)
- [Precision](src/detection_utils/metrics.py#L6) and [recall](src/detection_utils/metrics.py#L68) calculations
- [Computing targets](src/detection_utils/boxes.py#L60) for training a detector given a set of ground-truth objects
- [Non-maximum suppression](src/detection_utils/boxes.py#L141)
- [Coordinate transform utilities](src/detection_utils/boxes.py#L206)
- [Focal loss in PyTorch](src/detection_utils/pytorch.py#L5)

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
$ git clone https://github.com/davidmascharka/test.git && cd detection-utils
$ pip install .
```
## Contributing
If you're interested in contributing to this library, a good place to start is by browsing for [open
issues](https://github.com/davidmascharka/detection-utils/issues) to see if there are
desired features waiting for implementation. Please see the [contributors file](CONTRIBUTING.md) before you submit a
pull request.

## Copyright
DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.

This material is based upon work supported by the Assistant Secretary of Defense for Research and Engineering under Air
Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations
expressed in this material are those of the author(s) and do not necessarily reflect the views of the Assistant
Secretary of Defense for Research and Engineering.

© 2019 Massachusetts Institute of Technology.

MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
