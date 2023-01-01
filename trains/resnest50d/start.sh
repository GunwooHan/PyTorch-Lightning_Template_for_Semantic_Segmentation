python train.py --gpus=1 \
 --train_data_dir ~/codes/datasets/TianChiBuilding/train \
 --name resnest50dupp \
 --model ResNeSt50dUnetPP \
 --precision 16 \
 --buildingSegTransform True \
 --batch_size 24 \
 --epochs 24 \
 --learning_rate 0.0001 \
 --optimizer adamp \
#  --img_size 480

