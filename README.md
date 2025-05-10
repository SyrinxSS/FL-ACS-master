# FL-ACS: Achieving High Model Accuracy and Robustness at low Coreset Selection Ratios

#### Experiment on Image Classification Datasets

##### On CIFAR-10:

```
bash train_cifar10.sh
```

##### On CIFAR-100:

```
bash train_cifar100.sh
```

##### On SVHN Core:

```
bash train_svhn_core.sh
```

##### On tiny-ImageNet:

Download tiny-ImageNet: http://cs231n.stanford.edu/tiny-imagenet-200.zip to **data** and unzip.

```
bash train_tinyimagenet.sh
```

Change the **s** argument to the selection ratio you want.

Change the **arch** argument to the model you want.

#### Experiments on Complex Real-World Images

##### Class-imbalanced CIFAR-10:

```python
export CUDA_VISIBLE_DEVICES=0
python main.py cifar10_i_results -R 20 -s 0.1 -b 128 --dataset cifar10-i --lr 0.1 --epochs 200
```

##### Corrupted CIFAR-10:

Firstly, corrupt CIFAR-10 images and save. You can change corrupt ratio at Line 19.

```
cd robust
python corrupt-cifar10.py
```

Then, train models on CIFAR-10-C.

```
export CUDA_VISIBLE_DEVICES=0
python main.py cifar10_c_results -R 20 -s 0.1 -b 128 --dataset cifar10-c --lr 0.1 --epochs 200
```

##### Corrupted CIFAR-100:

Firstly, corrupt CIFAR-100 images and save.

```
cd robust
python corrupt-cifar.py
```

Then, train models on CIFAR-100-C.

```
export CUDA_VISIBLE_DEVICES=0
python main.py cifar100_c_results -R 20 -s 0.1 -b 128 --dataset cifar100-c --lr 0.1 --epochs 200
```

##### Label-Noisy CIFAR-10:

Firstly, add label noise to CIFAR-10 images and save. You can change label-noisy ratio at Line 11.

```
cd robust
python noisy-cifar10.py
```

Then, train models on CIFAR-10-N.

```
export CUDA_VISIBLE_DEVICES=0
python main.py cifar10_n_results -R 20 -s 0.1 -b 128 --dataset cifar10-n --lr 0.1 --epochs 200
```

##### Label-Noisy CIFAR-100:

Firstly,  add label noise to CIFAR-100 images and save. You can change label-noisy ratio at Line 11.

```
cd robust
python noisy-cifar.py
```

Then, train models on CIFAR-100-N.

```
export CUDA_VISIBLE_DEVICES=0
python main.py cifar100_n_results -R 20 -s 0.1 -b 128 --dataset cifar100-n --lr 0.1 --epochs 200
```

Change the **s** argument to the selection ratio you want.

Change the **arch** argument to the model you want.