import os
import sys
import inspect

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), '../../')))
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), './')))
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

from utils import train_test_split

import config as cfg
from utils import kaiming_init

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--train_data_dir', type=str, default='data/custom')
parser.add_argument('--test_data_dir', type=str, default='data/val')


parser.add_argument('--precision', type=int, default=32)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--num_workers', type=int, default=24)
parser.add_argument('--project', type=str, default='sikseki_segmentation_lv1')
parser.add_argument('--name', type=str, default='fcn_resnet50')
parser.add_argument('--model', type=str, default='Unet')
parser.add_argument('--encoder', type=str, default='resnet50')

# 학습 관련 설정

parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--kfold', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--optimizer', type=str, default='adamp')
parser.add_argument('--scheduler', type=str, default='reducelr')
# gpus
parser.add_argument('--gpus', type=int, default=1)
# buildingSegTransform
parser.add_argument('--buildingSegTransform', type=bool, default=True)
args = parser.parse_args()

if __name__ == '__main__':
    pl.seed_everything(args.seed)
    all_imgs = glob.glob(os.path.join(args.train_data_dir, '*.jpg'))
    all_masks = [x.replace('train', 'mask') for x in all_imgs]
    print(len(all_imgs), len(all_masks))
    
    # spilt train, val
    # train_images, val_images, train_masks, val_masks = train_test_split(all_imgs, all_masks, test_size=0.2, random_state=args.seed)
    # print(f'train_images: {len(train_images)}, val_images: {len(val_images)}')
    
    model = cfg.MODEL_INTERFACE(args, encoder='timm-resnest26d-addgsop')
    # model.apply(kaiming_init)
        
    kf = KFold(n_splits=args.kfold)
    print(args.kfold)
    for idx, (train_index, val_index) in enumerate(kf.split(X=all_imgs)):
        print(f'Fold: {idx}', 'train_index: ', len(train_index), 'val_index: ', len(val_index))
        wandb_logger = WandbLogger(project=args.project, group=args.name, name=f'{args.name}_fold{idx + 1:02d}')
        checkpoint_callback = ModelCheckpoint(
            monitor="val/jac_idx",
            dirpath="checkpoints",
            filename=f"{args.name}_fold{idx + 1:02d}_" + "{val/jac_idx:.4f}",
            save_top_k=3,
            mode="max",
            # save_weights_only=True
        )
        early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=300, verbose=True,
                                            mode="min")

        train_transform, val_transform = make_transform(args)

        train_ds = SegmentationDataset(np.array(all_imgs)[train_index], np.array(all_masks)[train_index], train_transform)
        train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, shuffle=True,
                                                       drop_last=True, pin_memory=True)

        val_ds = SegmentationDataset(np.array(all_imgs)[val_index], np.array(all_masks)[val_index], train_transform)
        val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
                                                     num_workers=args.num_workers, pin_memory=True)

        # test_ds = SegmentationDataset(test_images, test_masks, val_transform)
        # test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=1,
        #                                               num_workers=args.num_workers)
        from pytorch_lightning.strategies.ddp import DDPStrategy
        trainer = pl.Trainer(accelerator='gpu',
                            #  devices=1,
                             gpus=[0, 1],
                             precision=args.precision,
                             max_epochs=args.epochs,
                             log_every_n_steps=50,
                            #  amp_backend="apex",
                            #  auto_lr_find=True,
                            #  auto_scale_batch_size="binsearch",
                             strategy="ddp_find_unused_parameters_false",
                            #  strategy="cuda",
                             # num_sanity_val_steps=0,
                             # limit_train_batches=5,
                             # limit_val_batches=1,
                             logger=wandb_logger,
                             callbacks=[checkpoint_callback, early_stop_callback]
                             )
        
        # set broadcast_buffers=True
        # trainer.tune(model)
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        # trainer.test(dataloaders=test_dataloader)
        wandb.finish()
