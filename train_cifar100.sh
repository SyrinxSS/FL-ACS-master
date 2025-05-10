export CUDA_VISIBLE_DEVICES=0
python main.py cifar100_results -R 20 -s 0.1 -b 128 --dataset cifar100 --lr 0.1 --epochs 200
