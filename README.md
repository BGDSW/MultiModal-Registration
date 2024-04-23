# MultiModal-Registration

This is code for **Self-Supervised Structure-Preserved Image Registration Framework for Multimodal Retinal Images** https://ieeexplore.ieee.org/abstract/document/10390687

If you want to train your model, you need to add your dataset like this

```
DataFolder
    |-trainA
    |   |-imgA000.png
    |   |-imgA001.png
    |   ......
    |
    |-trainB
    |   |-imgB000.png
    |   |-imgB001.png
    |   ......
    |
    |-testA
    |   |......
    |
    |_testB
        |.....
```

Then you run

```
python train_unet_STN.py --dataroot ./DataFolder --name unet_STN23 --model unet_STN23 --n_epochs_decay 400 --save_epoch_freq 50 --display_freq 223  --unsupervised --lambda_depth 0.05 --depth_sum True
```

If you want to do test, you could run

```
python test.py --dataroot ./DataFolder --name unet_STN23 --model unet_STN23 --unsupervised --epoch 500
```

You can check performance in our paper
