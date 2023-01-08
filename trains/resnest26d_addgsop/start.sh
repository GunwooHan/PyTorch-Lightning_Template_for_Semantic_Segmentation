python train.py --gpus=3 \
 --train_data_dir ~/codes/datasets/TianChiBuilding/train \
 --name resnest14dupp \
 --name ResNeSt26dAddGSoPUnetPP \
 --model ResNeSt26dAddGSoPUnetPP \
 --precision 32 \
 --buildingSegTransform True \
 --batch_size 32 \
 --epochs 24 \
 --learning_rate 0.001 \
 --optimizer sgd \
 --scheduler cosineanneal

