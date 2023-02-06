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
from models.unet_inferface import UnetModel
from utils import train_test_split
import cv2
import config as cfg

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_img_dir', type=str, default='/home/aieson/codes/datasets/buildingSegDataset/preds/pred_img')
    parser.add_argument('--pred_mask_name', type=str, default='target_mask')
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--model_name', type=str, default=f'{cfg.N_NAME}')
    parser.add_argument('--buildingSegTransform', type=bool, default=True)
    parser.add_argument('--predict_result_dir', type=str, default='')
    return parser.parse_args()

def get_all_images(path):
    
    return glob.glob(os.path.join(path, '*.jpg'))

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(args):
    model = cfg.MODEL_INTERFACE.load_from_checkpoint(args.model_file)
    model.eval()
    imgs = get_all_images(args.pred_img_dir)
    print(imgs)
    dataset = SegmentationDataset(imgs, transform=make_transform(args)[1])
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, img in enumerate(loader):
        print(i)
        print(img.shape)
        pred = model(img)
        # 输出shape为 [1, 2, 512, 512]
        # 合并双channel 为单channel图片
        pred = torch.argmax(pred, dim=1)
        # print(pred.shape)
        # 保存图片
        pred = pred.squeeze(0).cpu().numpy()
        # print(pred.shape)
        # 保存图片
        save_path = os.path.join(args.predict_result_dir, f'{i}.jpg') \
                    if args.predict_result_dir != '' \
                    else f'{os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), "predict_result", str(i) + ".jpg")}'
        # 获取父目录
        make_dir(os.path.dirname(save_path))
        cv2.imwrite(save_path, pred * 255)
        # 保存原图
        cv2.imwrite(save_path.replace('.jpg', '_origin.jpg'), cv2.imread(imgs[i]))
        print(imgs[i].replace('pred_img', 'target_mask'))
        cv2.imwrite(save_path.replace('.jpg', '_mask.jpg'), cv2.imread(imgs[i].replace('pred_img', 'target_mask')))
        
    
if __name__ == '__main__':
    args = init_args()
    main(args)