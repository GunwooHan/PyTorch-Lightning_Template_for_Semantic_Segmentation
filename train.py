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
parser.add_argument('--train_data_dir', type=str, default='data/custom')
parser.add_argument('--test_data_dir', type=str, default='data/val')

parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--precision', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--name', type=str, default='custom_model')

# 모델 관련 설정
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--project', type=str, default='sikseki_segmentation_lv1')
parser.add_argument('--name', type=str, default='fcn_resnet50')
parser.add_argument('--model', type=str, default='Unet')
parser.add_argument('--encoder', type=str, default='resnet50')

# 학습 관련 설정

parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--kfold', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--optimizer', type=str, default='adamp')
parser.add_argument('--scheduler', type=str, default='reducelr')
# parser.add_argument('--loss', type=str, default='ce')

# augmentation 관련 설정
parser.add_argument('--crop_image_size', type=int, default=512)
parser.add_argument('--ShiftScaleRotateMode', type=int, default=4)
parser.add_argument('--ShiftScaleRotate', type=float, default=0.2)
parser.add_argument('--HorizontalFlip', type=float, default=0.2)
parser.add_argument('--VerticalFlip', type=float, default=0.2)
args = parser.parse_args()

args = parser.parse_args()

if __name__ == '__main__':
    pl.seed_everything(args.seed)
    train_images = np.array(sorted(glob.glob(os.path.join(args.train_data_dir, 'images', '*'))))
    train_masks = np.array(sorted(glob.glob(os.path.join(args.train_data_dir, 'masks', '*'))))

    test_images = np.array(sorted(glob.glob(os.path.join(args.test_data_dir, 'images', '*'))))
    test_masks = np.array(sorted(glob.glob(os.path.join(args.test_data_dir, 'masks', '*'))))

    kf = KFold(n_splits=args.kfold)
    for idx, (train_index, val_index) in enumerate(kf.split(X=train_images)):
        wandb_logger = WandbLogger(project=args.project, group=args.name, name=f'{args.name}_fold{idx + 1:02d}')
        checkpoint_callback = ModelCheckpoint(
            monitor="val/jaccard_index_value",
            dirpath="checkpoints",
            filename=f"{args.name}_fold{idx + 1:02d}_" + "{val/jaccard_index_value:.4f}",
            save_top_k=3,
            mode="max",
            # save_weights_only=True
        )
        early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=50, verbose=True,
                                            mode="min")

        train_transform, val_transform = make_transform(args)
        model = PlSegmentationModel(args)

        train_ds = SegmentationLv1Dataset(train_images[train_index], train_masks[train_index], train_transform)
        train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, shuffle=True,
                                                       drop_last=True)

        val_ds = SegmentationLv1Dataset(train_images[val_index], train_masks[val_index], train_transform)
        val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
                                                     num_workers=args.num_workers)

        test_ds = SegmentationLv1Dataset(test_images, test_masks, val_transform)
        test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=1,
                                                      num_workers=args.num_workers)

        trainer = pl.Trainer(accelerator='gpu',
                             devices=args.gpus,
                             precision=args.precision,
                             max_epochs=args.epochs,
                             log_every_n_steps=1,
                             # strategy='ddp',
                             # num_sanity_val_steps=0,
                             # limit_train_batches=5,
                             # limit_val_batches=1,
                             logger=wandb_logger,
                             callbacks=[checkpoint_callback, early_stop_callback])

        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        # trainer.test(dataloaders=test_dataloader)
        wandb.finish()