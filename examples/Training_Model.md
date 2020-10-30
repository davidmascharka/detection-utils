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
from pytorch_lightning.callbacks import ModelCheckpoint
```

```python
ls
```

```python
checkpointer = ModelCheckpoint(
    filepath="./bogs/",
    monitor="ap+ar",
    save_top_k=5,
    save_last=True,
    mode="max",
    period=3,
)
```

```python
model = ShapeDetectionModel(data_experiment_path="./data")

assert model.data_path is not None

trainer = pl.Trainer(gpus=1, max_epochs=2, checkpoint_callback=checkpointer, default_root_dir="./bogs")
trainer.fit(model)
```

```python
# 29 - no batchnorm
# 30 - batchnorm pre-relu
# 31 - batchnorm post-relu
# 32 - batchnorm post-pooling
```

```python
import numpy as np
```

```python
x = np.identity(3)
x[0, 2] = 5
x.argmax(1)
```

```python

```

```python
conf = np.zeros((4, 4), dtype=np.int32)
predictions = np.array([0, 1, 1, 0, 2, 3, 3, 0])
actual =      np.array([0, 1, 2, 3, 0, 3, 3, 0])

np.add.at(conf, (predictions, actual), 1)
```

```python

```

```python
conf
```
