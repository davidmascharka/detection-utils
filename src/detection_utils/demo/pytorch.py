from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch as tr
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam

from ..pytorch import softmax_focal_loss


def loss(
    class_predictions: Tensor,
    regression_predictions: Tensor,
    class_targets: Tensor,
    regression_targets: Tensor,
) -> Tuple[Tensor, Tensor]:

    """
    Computes the classification and regression
      smooth L1 (Huber) loss for regression (only on foreground anchor boxes)
      softmax focal loss for classification (excluding ignore anchor boxes)

    Parameters
    ----------
    class_predictions : Tensor, shape-(N, K, num-class)
    regression_predictions : Tensor, shape-(N, K, 4)
    class_targets : Tensor, shape-(N, K)
    regression_targets : Tensor, shape-(N, K, 4)

    Returns
    -------
    classification_loss, regression_loss: Tuple[Tensor, Tensor], shape-() shape-()
        The mean classification loss and regression loss, respectively.

    Notes
    -----
    `N` is the batch size. `K` is the number of anchor boxes associated with
    each image.
    """
    # shape-(N*K,)
    class_targets = class_targets.reshape(-1)

    # shape-(N*K, 4)
    regression_targets = regression_targets.reshape(-1, 4)

    # shape-(N*K, num-class)
    class_predictions = class_predictions.reshape(-1, class_predictions.shape[-1])

    # shape-(N*K, 4)
    regression_predictions = regression_predictions.reshape(-1, 4)

    mask = tr.squeeze(class_targets > 0)
    num_foreground = mask.sum().item()
    if mask.numel() > 0:
        regression_loss = F.smooth_l1_loss(
            regression_predictions[mask], regression_targets[mask]
        )
    else:
        regression_loss = tr.tensor(0).float()

    mask = tr.squeeze(class_targets > -1)

    # the sum of focal loss terms is normalized by the number
    # of anchors assigned to a ground-truth box
    classification_loss = (
        softmax_focal_loss(
            class_predictions[mask],
            class_targets[mask],
            alpha=0.25,
            gamma=2,
            reduction="sum",
        )
        / num_foreground
    )

    return classification_loss, regression_loss


class ShapeDetectionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 20, 3, padding=1)
        self.conv3 = nn.Conv2d(20, 30, 3, padding=1)
        self.conv4 = nn.Conv2d(30, 40, 3, padding=1)

        # background / rectangle / triangle / circle
        self.classification = nn.Conv2d(40, 4, 1)
        self.regression = nn.Conv2d(40, 4, 1)

        for layer in (
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.classification,
            self.regression,
        ):
            nn.init.xavier_normal_(layer.weight, np.sqrt(2))
            nn.init.constant_(layer.bias, 0)

        nn.init.constant_(
            self.classification.bias[0], -4.6
        )  # rougly -log((1-π)/π) for π = 0.01

    def forward(self, imgs: Tensor) -> Tuple[Tensor, Tensor]:
        """"
        Computes the classification scores and bounding box regression associated
        with each anchor box of each image.

        Parameters
        ----------
        imgs : Tensor, shape-(N, 3, H, W)
            A batch of N images.

        Returns
        -------
        classifications, regressions : Tuple[Tensor, Tensor]
            shape-(N, K, N_class), shape-(N, K, 4)
            For each of N images in the batch, returns the classification scores
            and bbox regressions associated with each of the K anchor boxes associated
            with that image.

        Notes
        -----
        The anchor boxes are flattened in row-major order"""
        imgs = F.max_pool2d(F.relu(self.conv1(imgs)), 2)
        imgs = F.max_pool2d(F.relu(self.conv2(imgs)), 2)
        imgs = F.max_pool2d(F.relu(self.conv3(imgs)), 2)
        imgs = F.max_pool2d(F.relu(self.conv4(imgs)), 2)

        # (N, num-classes, R, C) -> (N, R, C, num-classes)
        classifications = self.classification(imgs).permute(0, 2, 3, 1)
        # (N, R, C, num-classes) -> (N, R*C, num-classes)
        classifications = classifications.reshape(
            imgs.shape[0], -1, classifications.shape[-1]
        )
        # (N, 4, R, C) -> (N, R, C, 4)
        regressions = self.regression(imgs).permute(0, 2, 3, 1)
        # (N, R, C, 4) ->  (N, R*C, 4)
        regressions = regressions.reshape(imgs.shape[0], -1, 4)
        return classifications, regressions

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        imgs, class_targets, bbox_targets = batch
        class_predictions, regression_predictions = self(imgs)
        total_cls_loss, total_reg_loss = loss(
            class_predictions, regression_predictions, class_targets, bbox_targets,
        )
        return total_cls_loss + total_reg_loss

    def validation_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        imgs, class_targets, bbox_targets = batch
        class_predictions, regression_predictions = self(imgs)
        total_cls_loss, total_reg_loss = loss(
            class_predictions, regression_predictions, class_targets, bbox_targets,
        )
        self.log("val_loss", total_cls_loss + total_reg_loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=5e-4)
