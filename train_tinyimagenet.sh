export CUDA_VISIBLE_DEVICES=0
python main.py tinyimagenet_results -R 15 -s 0.5 -b 128 --dataset tinyimagenet --lr 1e-1 --epochs 90 --wd 1e-4 --arch resnet18
