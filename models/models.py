import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from adamp import AdamP
from torchmetrics.functional import jaccard_index



class SegmentationModel(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()
        self.model = smp.__dict__[args.model](
            encoder_name=args.encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,  # model output channels (number of classes in your dataset)
        )
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
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, mode="max", verbose=True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/jaccard_index_value"}

        elif self.args.scheduler == "cosineanneal":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5,
                                                                 last_epoch=-1, verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, train_batch, batch_idx):
        image, mask = train_batch
        mask = mask.long()

        outputs = self.model(image)
        loss = self.criterion(outputs, mask)
        jaccard_index_value = jaccard_index(outputs.argmax(dim=1), mask, num_classes=2)

        self.log('train/loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('train/jaccard_index_value', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)

        return {"loss": loss, "jaccard_index_value": jaccard_index_value}

    def validation_step(self, val_batch, batch_idx):
        image, mask = val_batch
        mask = mask.long()

        outputs = self.model(image)
        loss = self.criterion(outputs, mask)
        jaccard_index_value = jaccard_index(outputs.argmax(dim=1), mask, num_classes=2)

        self.log('val/loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('val/jaccard_index_value', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)

        return {"loss": loss, "jaccard_index_value": jaccard_index_value}

    def test_step(self, test_batch, batch_idx):
        image, mask = test_batch
        mask = mask.long()

        outputs = self.model(image)
        loss = self.criterion(outputs, mask)
        jaccard_index_value = jaccard_index(outputs.argmax(dim=1), mask, num_classes=2)

        self.log('test/loss', loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log('test/jaccard_index_value', jaccard_index_value, on_epoch=True, on_step=False, prog_bar=True,
                 sync_dist=True)

        return {"loss": loss, "jaccard_index_value": jaccard_index_value}
