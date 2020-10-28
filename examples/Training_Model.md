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

# Training Model

```python
from detection_utils.demo.pytorch import ShapeDetectionModel
import pytorch_lightning as pl
```

```python
model = ShapeDetectionModel(data_experiment_path="./data")

assert model.data_path is not None

trainer = pl.Trainer(gpus=1, max_epochs=150)
trainer.fit(model)
```
