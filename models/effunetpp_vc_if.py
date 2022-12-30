from pyrsistent import m
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from adamp import AdamP
from torchmetrics.functional import jaccard_index, dice_score, f1_score, precision, recall, accuracy
import segmentation_models_pytorch as smp

import os
import sys
import inspect

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), '../')))

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), '../..')))

from losses import FocalLoss
from models.att_block import VCModule

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2,-1)):

        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims
    
    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc
    
bce_fn = nn.BCEWithLogitsLoss()
dice_fn = SoftDiceLoss()

def loss_fn(y_pred, y_true):
    bce = bce_fn(y_pred, y_true)
    dice = dice_fn(y_pred.sigmoid(), y_true)
    return 0.8*bce+ 0.2*dice
    # return bce

# n_loss = FocalLoss(gamma=2)
# n_loss = nn.NLLLoss()


# def loss_fn(outputs, targets):
#     return  n_loss(outputs, targets)


def mask_onehot(masks):
    # batch_size, h, w
    masks = masks.long()
    masks_onehot = torch.zeros(masks.size(0), masks.max() + 1, masks.size(1), masks.size(2)).to(masks.device)
    masks_onehot = masks_onehot.scatter_(1, masks.unsqueeze(1), 1)
    return masks_onehot
    


class EffUnetPPModel_VC(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()
        self.model1 = smp.UnetPlusPlus(encoder_name='efficientnet-b7', encoder_weights='imagenet', in_channels=3, classes=64)
        self.model2 = VCModule(64, 1)
        # self.model2 = nn.Sequential()
        
        # 模型运行到倒数第二层
        self.args = args
        self.criterion = loss_fn

    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
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
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/jac_idx"}

        elif self.args.scheduler == "cosineanneal":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5,
                                                                 last_epoch=-1, verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, train_batch, batch_idx):
        image, mask = train_batch
        # Mask 增加一个维度
        mask = mask.long()

        outputs = self.model2(self.model1(image))
        loss = self.criterion(outputs, mask.unsqueeze(1).float())
        pre_label = outputs.sigmoid()
        jaccard_index_value = jaccard_index(pre_label, mask, num_classes=2)
        f1 = f1_score(pre_label, mask, multiclass=True, num_classes=2, mdmc_average = 'global')
        acc = accuracy(pre_label, mask,multiclass=True, num_classes=2, mdmc_average = 'global')

        self.log('train/loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('train/jac_idx', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)
        self.log('train/f1', f1, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('train/acc', acc, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        

        return {"loss": loss, "jac_idx": jaccard_index_value, "f1": f1, "acc": acc}

    def validation_step(self, val_batch, batch_idx):
        image, mask = val_batch
        mask = mask.long()
        
        outputs = self.model2(self.model1(image))
        loss = self.criterion(outputs, mask.unsqueeze(1).float())
        pre_label = outputs.sigmoid()
        jaccard_index_value = jaccard_index(pre_label, mask, num_classes=2)
        f1 = f1_score(pre_label, mask, multiclass=True, num_classes=2, mdmc_average = 'global')
        acc = accuracy(pre_label, mask,multiclass=True, num_classes=2, mdmc_average = 'global')


        self.log('val/loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('val/jac_idx', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)
        self.log('val/f1', f1, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('val/acc', acc, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        

        return {"loss": loss, "jac_idx": jaccard_index_value, "f1": f1, "acc": acc}

    def test_step(self, test_batch, batch_idx):
        image, mask = test_batch
        mask = mask.long()

        outputs = self.model2(self.model1(image))
        loss = self.criterion(outputs, mask.unsqueeze(1).float())
        pre_label = outputs.sigmoid()
        jaccard_index_value = jaccard_index(pre_label, mask, num_classes=2)
        f1 = f1_score(pre_label, mask, multiclass=True, num_classes=2, mdmc_average = 'global')
        acc = accuracy(pre_label, mask,multiclass=True, num_classes=2, mdmc_average = 'global')

        
        self.log('test/loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('test/jac_idx', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)
        self.log('test/f1', f1, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('test/acc', acc, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
         

        return {"loss": loss, "jac_idx": jaccard_index_value, "f1": f1, "acc": acc}

# import segmentation_models_pytorch as smp

if __name__ == '__main__':
    model_effupp = smp.UnetPlusPlus(encoder_name='efficientnet-b7', encoder_weights='imagenet', in_channels=3, classes=128)
    # 模型运行到倒数第二层
    vcmodel = VCModule(128, 1)
    model = nn.Sequential(model_effupp, vcmodel)
    
    
    
    a = torch.randn(1, 3, 256, 256)
    b = model(a)
    print(b.shape)