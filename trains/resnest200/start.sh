python train.py --gpus=3 \
 --train_data_dir ~/codes/datasets/buildingSegDataset/train \
 --name resnest200eupp \
 --model ResNeSt200eUnetPP \
 --precision 16 \
 --buildingSegTransform True \
 --batch_size 12 \
 --epochs 48 \
 --learning_rate 0.0001 \
 --optimizer adamp \
 --img_size 480

