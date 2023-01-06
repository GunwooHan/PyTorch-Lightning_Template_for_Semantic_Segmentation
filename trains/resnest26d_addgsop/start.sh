python train.py --gpus=3 \
 --train_data_dir ~/codes/datasets/buildingSegDataset/train \
 --name ResNeSt26dAddGSoPUnetPP \
 --model ResNeSt26dAddGSoPUnetPP \
 --precision 16 \
 --buildingSegTransform True \
 --batch_size 16 \
 --epochs 24 \
 --learning_rate 0.00008 \
 --optimizer adamw \
 --img_size 480

