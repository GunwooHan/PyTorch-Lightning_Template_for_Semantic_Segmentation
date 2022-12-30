python train.py --gpus=3 --train_data_dir ~/codes/datasets/buildingSegDataset/train --name my_effunetpp_v2 --model My_EffUnetPP_V2 --precision 16 --buildingSegTransform True --batch_size 16 --epochs 40 --learning_rate 0.0001 --optimizer adamw --img_size 480

