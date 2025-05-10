export CUDA_VISIBLE_DEVICES=0
python main.py cifar10_results -R 20 -s 0.1 -b 128 --dataset cifar10 --lr 0.1 --epochs 200