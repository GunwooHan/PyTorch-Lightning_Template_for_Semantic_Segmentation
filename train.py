import os
import glob
import argparse

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from datasets import SegmentationDataset
from transforms import make_transform
from models import SegmentationModel

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--train_datadir', type=str, default='data/custom')
parser.add_argument('--val_datadir', type=str, default='data/val')

parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--precision', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--name', type=str, default='custom_model')

# parser.add_argument('--kfold', type=str, default='custom_model')
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--auto_batch_size', type=bool, default=False, help='Search batch size')
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--optimizer', type=str, default='adamp')
parser.add_argument('--scheduler', type=str, default='reducelr')
parser.add_argument('--loss', type=str, default='ce')

parser.add_argument('--Blur', type=float, default=0)
parser.add_argument('--RandomBrightnessContrast', type=float, default=0)
parser.add_argument('--HueSaturationValue', type=float, default=0)
parser.add_argument('--RGBShift', type=float, default=0)
parser.add_argument('--RandomGamma', type=float, default=0)
parser.add_argument('--HorizontalFlip', type=float, default=0)
parser.add_argument('--VerticalFlip', type=float, default=0)
parser.add_argument('--ImageCompression', type=float, default=0)
parser.add_argument('--ShiftScaleRotate', type=float, default=0)
parser.add_argument('--ShiftScaleRotateMode', type=int, default=4) # Constant, Replicate, Reflect, Wrap, Reflect101
parser.add_argument('--Downscale', type=float, default=0)
parser.add_argument('--GridDistortion', type=float, default=0)
parser.add_argument('--MotionBlur', type=float, default=0)
parser.add_argument('--RandomResizedCrop', type=float, default=0)
parser.add_argument('--CLAHE', type=float, default=0)

args = parser.parse_args()

if __name__ == '__main__':
    pl.seed_everything(args.seed)
    images = np.array(glob.glob(os.path.join(args.train_datadir, 'images', '*')))
    masks = np.array(glob.glob(os.path.join(args.train_datadir, 'masks', '*')))

    kf = KFold(n_splits=5)
    for idx, (train_index, val_index) in enumerate(kf.split(X=images)):
        wandb_logger = WandbLogger(project=f'{args.name}', group=args.name, name=f'{args.name}_fold{idx + 1:02d}')
        checkpoint_callback = ModelCheckpoint(
            monitor="val/mIoU",
            dirpath="saved",
            filename=f"{args.name}_fold{idx + 1:02d}_" + "{val/mIoU:.4f}",
            save_top_k=3,
            mode="max",
            # save_weights_only=True
        )
        early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=5, verbose=True,
                                            mode="min")

        train_transform, val_transform = make_transform(args)
        model = SegmentationModel(args)

        train_ds = SegmentationDataset(images[train_index], masks[train_index], train_transform)
        train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, shuffle=True,
                                                       drop_last=True)

        val_ds = SegmentationDataset(images[val_index], masks[val_index], val_transform)
        val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
                                                     num_workers=args.num_workers)

        trainer = pl.Trainer(gpus=args.gpus,
                             precision=args.precision,
                             max_epochs=args.epochs,
                             log_every_n_steps=1,
                             strategy='ddp',
                             # num_sanity_val_steps=0,
                             limit_train_batches=5,
                             limit_val_batches=1,
                             logger=wandb_logger,
                             stochastic_weight_avg=True,
                             callbacks=[checkpoint_callback, early_stop_callback])

        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        wandb.finish()