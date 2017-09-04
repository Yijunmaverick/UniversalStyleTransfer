# UniversalStyleTransfer
Torch implementation of our [paper](https://arxiv.org/pdf/1705.08086.pdf) on universal style transfer. For academic use only.

## Prerequisites

- Linux
- NVIDIA GPU + CUDA CuDNN
- Torch 
- Pretrained encoder and decoder [models](https://drive.google.com/open?id=0B8_MZ8a8aoSeWm9HSTdXNE9Eejg) for image reconstruction only (download and put them under models/)

## Style transfer

- For a single pair test:

```
th test_wct.lua -content YourContentImagePath -style YourStyleImagePath -alpha YourStyleWeight
```

- For large numbers of pair test:

```
th test_wca.lua -contentDir YourContentImageDir -styleDir YourStyleImageDir -alpha YourStyleWeight
```

By default, we perform WCT (whitening and coloring transform) on conv1-5 features. 


## Texture synthesis

```
th test_wct.lua -style YourStyleImagePath -synthesis 1 
```


## Spatial control

Style1 for foreground (mask=1), Style2 for background (mask=0), provided a binary mask

<img src='figs/p2.jpg' width=800>

```
th test_wct_SpatialControl_withMask.lua -content YourConentPath -style YourStylePath1,YourStylePath2 -mask YourBinaryMaskPath
```

## Swap on conv5

We provide a parameter "-swap5 1" to perform swap operation on conv5 features. 

```
th test_wct.lua -content YourContentImagePath -style YourStyleImagePath -swap5 1
```

<img src='figs/p1.jpg' width=840>


## Note

- In theory, the covariance matrix of whitened features should be Identity. In practise, it is not because we need to eliminate some extremely small eigen values (e.g., <1e-10) or add a small constant (e.g., 1e-7) to all eigen values in order to perform the inverse operation (D^-1/2) in the whitening.

- To save memory for testing image of large size, we need to often load and delete model. So in our code, for the transferring on each content/style pair, we need to reload the model.

- For a GPU of memory ~12G, it is suggested that the contentSize and styleSize are all less than 900 (800 recommended for the largest size).

- We found that using "CUDA_VISIBLE_DEVICES=X" is better than using "-gpu X" as the former choice will guarantee that all weights/gradients/input will be located on the same GPU.

## Citation


## Acknowledgement

We express gratitudes to the great work [AdaIN](https://github.com/xunhuang1995/AdaIN-style) and [Style-swap](https://github.com/rtqichen/style-swap) as we benefit a lot from both their paper and codes.
