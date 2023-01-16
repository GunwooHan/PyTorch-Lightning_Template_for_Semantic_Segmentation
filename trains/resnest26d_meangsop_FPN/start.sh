python train.py --gpus=3 \
 --train_data_dir ~/codes/datasets/TianChiBuilding/train \
 --name ResNeSt26dMeanGSoPFPN \
 --model ResNeSt26dMeanGSoPFPN \
 --precision 32 \
 --buildingSegTransform True \
 --batch_size 32 \
 --epochs 64 \
 --learning_rate 0.0005 \
 --optimizer adamp \
 --kfold 2 \
