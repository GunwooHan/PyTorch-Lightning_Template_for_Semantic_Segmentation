# PyTorch Lightning Template for Semantic Segmentation
- Getting start for the Semantic Segmentation Competition quickly

## Requirements
- Python >= 3.7 (3.8 recommended)
- Pytorch >= 1.7 (conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch)
- pytorch_lightning
- tqdm
- albumentations
- wandb
- adamp
- opencv-python

## Folder Structure

```
      │
      ├── data/ - abstract base classes
      │   └── train/
      │   │     ├── images/
      │   │     └── masks/
      │   └── val/
      │         ├── images/
      │         └── masks/
      │ 
      ├── saved/ - abstract base classes
      │ 
      ├── datasets.py - main script to start training
      ├── models.py - evaluation of trained model
      ├── test.py - evaluation of trained model
      ├── train.py - evaluation of trained model
      ├── transforms.py - evaluation of trained model
      └── utils.py - evaluation of trained model
```

## Usage
There are Basic features of `Trainer` of pytorch lightning  
Some features what i fequently used are made arguments `args`

### Using Multiple GPU
If you want to train multi-GPU machine, you just add argument `--gpus`

```
    python train.py --gpus 4 
```
### Precision
If you want to other precisions, you just add argument `--precision`

```
    python train.py --precision 16 # 16, 32, 64, mixed available
```

### Data Augmnetation
Data Augmnetation is basically implemented by `Albumentations`  
If you want to add others, append Augumentation in `transforms.py`

```
    train_transform = []
    train_transform.append(A.Augmentation())
    
    ...

```
### Optimizer & Scheduler
I implemented 3 optimizers [`adam`, `adamw`, `adamp`] and 2 schedulers [`cosine annealing`, `reducelr`]
if you want to use other optimizer & scheduler, edit `configure_optimizers` function in `models.py`
When you use reducelr, essentialy implement `monitor`. 

```
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
```

## TODOs
- [ ] Implement resume training
- [ ] test.py implement
- [ ] add feature TTA
