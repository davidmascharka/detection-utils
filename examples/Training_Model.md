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
from pathlib import Path

from detection_utils.demo.pytorch import ShapeDetectionModel
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

import pickle
```

```python
model = ShapeDetectionModel(data_experiment_path="./data")
checkpointer = ModelCheckpoint(
    monitor="ap+ar",
    save_top_k=-1,
    save_last=True,
    mode="max",
    period=3,
)

trainer = pl.Trainer(
    gpus=1,
    max_epochs=150,
    checkpoint_callback=checkpointer,
)
trainer.fit(model)

logger = model.logger
save_dir = Path(logger.save_dir, logger.name, f"version_{logger.version}")

# save data for post-run visualization
with (save_dir / "confusion_matrices.pkl").open("wb") as f:
    pickle.dump(model.confusion_matrices, f)

with (save_dir / "val_img_1_info.pkl").open("wb") as f:
    pickle.dump(model.boxes_labels_scores, f)
```
