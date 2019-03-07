# Contributing
If you're interested in contributing to this library, a good place to start is by browsing for [open issues](https://github.com/davidmascharka/detection-utils/issues) to see if there are desired features waiting for implementation. Before you submit a pull request, make sure that:

- Tests are included (and passing)
- Your branch is up-to-date with master
- Your code is stylistically-consistent with the library
- Your code is well-documented

If these conditions are not met, your pull request will not be approved. If you have any doubts, we the maintainers are happy to work with you to craft an initial PR into something that fits with the library.

### Running Tests

To run tests on the code, make sure that you have hypothesis and pytest installed. Then run `pytest` from the base directory of this project. You should see an output like:

```
====================== test session starts =======================
platform linux -- Python 3.7.3, pytest-4.6.2, py-1.8.0, pluggy-0.12.0
hypothesis profile 'ci' -> deadline=None, database=DirectoryBasedExampleDatabase('/home/david/detection-utils/.hypothesis/examples')
rootdir: /home/david/detection-utils
plugins: cov-2.7.1, hypothesis-4.24.0
collected 48 items                                               

tests/test_boxes.py .........................              [ 52%]
tests/test_metrics.py ..............                       [ 81%]
tests/test_pytorch.py .........                            [100%]

=================== 48 passed in 7.02 seconds ====================
```

All your tests should be passing before you submit a pull request.

### Generating code-coverage reports

We strive to ensure that all our lines of code are covered by tests. In order to generate a coverage report, you'll need to install `pytest-cov` and run:

``` shell
$ NUMBA_DISABLE_JIT=1 pytest --cov-report term-missing --cov=detection_utils tests/
```

from the base directory of this project. The `NUMBA_DISABLE_JIT` environment variable is required in order to generate a coverage report for numba-jitted functions (such as [box_overlaps](https://github.com/davidmascharka/detection-utils/blob/master/src/detection_utils/boxes.py#L29)). This should produce an output like:

```
----------- coverage: platform linux, python 3.7.3-final-0 -----------
Name                              Stmts   Miss  Cover   Missing
---------------------------------------------------------------
src/detection_utils/__init__.py       3      0   100%
src/detection_utils/boxes.py         72      0   100%
src/detection_utils/metrics.py       32      0   100%
src/detection_utils/pytorch.py       11      0   100%
---------------------------------------------------------------
TOTAL                               118      0   100%
```

Ideally, your coverage should be 100% before you submit a pull request.
