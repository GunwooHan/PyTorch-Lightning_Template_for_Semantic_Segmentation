import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduelr
from adamp import AdamP
import pytorch_lightning as pl
from torchmetrics.functional import iou, accuracy

import torchvision

class SegmentationModel(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()
        self.model = torchvision.models.segmentation.fcn_resnet50(num_classes=15)
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamp':
            optimizer = AdamP(self.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)

        if self.args.scheduler == "reducelr":
            scheduler = lr_scheduelr.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, mode="max", verbose=True)
        elif self.args.scheduler == "cosineanneal":
            scheduler = lr_scheduelr.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5,
                                                                 last_epoch=-1, verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/mIoU"}

    def training_step(self, train_batch, batch_idx):
        image, mask = train_batch
        mask = mask.long()

        outputs = self.model(image)['out']
        loss = self.criterion(outputs, mask)
        iou_value = iou(outputs.argmax(dim=1), mask)
        acc_value = accuracy(outputs, mask)

        self.log('train/loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log('train/acc', acc_value, on_epoch=True, on_step=True, prog_bar=True)
        self.log('train/mIoU', iou_value, on_epoch=True, on_step=True, prog_bar=True)

        return {"loss": loss, "IoU": iou_value, "acc": acc_value}

    def validation_step(self, val_batch, batch_idx):
        image, mask = val_batch
        mask = mask.long()

        outputs = self.model(image)['out']
        loss = self.criterion(outputs, mask)
        iou_value = iou(outputs.argmax(dim=1), mask)
        acc_value = accuracy(outputs, mask)

        self.log('val/loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log('val/acc', acc_value, on_epoch=True, on_step=True, prog_bar=True)
        self.log('val/mIoU', iou_value, on_epoch=True, on_step=True, prog_bar=True)

        return {"loss": loss, "IoU": iou_value, "acc": acc_value}

    # def train_dataloader(self):
    #     return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.args.num_workers, shuffle=True)
    #
    # def val_dataloader(self):
    #     return torch.utils.data.DataLoader(self.val_data, batch_size=1, num_workers=self.args.num_workers)
