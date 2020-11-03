---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.0
  kernelspec:
    display_name: Python [conda env:.conda-raains]
    language: python
    name: conda-env-.conda-raains-py
---

# Our Model

We are training a "single-stage" detector, based on the [RetinaNet in
the paper "Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002).

## Single Stage Detection

Our model will make predictions over a regular, dense sampling of potential object locations (this set of dense samples will be known as our "anchor boxes").
This is in contrast to "two stage" detectors, like [Faster RCNN](https://arxiv.org/abs/1506.01497).

Two stage object detectors often operate first via a "region proposal stage", which generates a sparse set of candidate objects.
Then the second stage applies a classifier model to the candidates to classify them as a type of foreground object or as background, 





```python

```
