python train.py --gpus=3 \
 --train_data_dir ~/codes/datasets/buildingSegDataset/train \
 --name resnest26_mvcdupp \
 --model ResNeSt26d_MVCUnetPP \
 --precision 16 \
 --buildingSegTransform True \
 --batch_size 24 \
 --epochs 24 \
 --learning_rate 0.0001 \
 --optimizer adamw \
 --img_size 480

