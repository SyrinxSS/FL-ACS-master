export CUDA_VISIBLE_DEVICES=0
python main.py svhn_core_results -R 20 -s 0.5 -b 128 --dataset svhn_core --lr 0.1 --epochs 200