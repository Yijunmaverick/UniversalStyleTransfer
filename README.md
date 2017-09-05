# UniversalStyleTransfer
Torch implementation of our NIPS17 [paper](https://arxiv.org/pdf/1705.08086.pdf) on universal style transfer. For academic use only.

Universal style transfer aims to transfer any arbitrary visual styles to content images. As long as you can find your desired style images on web, you can edit your content image with different transferring effects. 


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
th test_wct.lua -contentDir YourContentImageDir -styleDir YourStyleImageDir -alpha YourStyleWeight
```

By default, we perform WCT (whitening and coloring transform) on conv1-5 features. Some transfer results and comparisons with existing methods are shown [here](https://drive.google.com/file/d/0B8_MZ8a8aoSed3RrcTBfM1hES3c/view).


## Texture synthesis

Though the style transfer arouses more interests, texture synthesis is the more essential problem and has more applications. By simplying filling the content image with noise (setting the parameter "-synthesis" as 1), our algorithm turns to synthsize the texture (or style). Different input noise leads to diverse synthesis results. Moreover, we can adjust the parameter "-styleSize" as a kind of scale control to synthsize different effects.

```
th test_wct.lua -style YourStyleImagePath -synthesis 1 -styleSize 512
```

<img src='figs/p4.jpg' width=800>


## Spatial control

Often times, the one-click global transfer is not what professinal artists want. Users prefer to transfer different styles to different regions in the content image. We provide an example of transferring two styles to the foreground and background respectively, i.e., Style I for foreground (mask=1), Style II for background (mask=0), provided a binary mask.

```
th test_wct_mask.lua -content YourConentPath -style YourStylePath1,YourStylePath2 -mask YourBinaryMaskPath
```

<img src='figs/p2.jpg' width=800>


## Swap on conv5

We also include the [Style-swap](https://github.com/rtqichen/style-swap) function in our algorithm but we perform the swap operation based on whitened features. For each whitened feature patch, we swap it with nearest whitened style patch. Please refer to their [paper](https://arxiv.org/pdf/1612.04337.pdf) for more details.

We provide a parameter "-swap5" to perform swap operation on conv5 features. The swap operation is computationally expensive as it is based on searching nearest patches. So we do not provide the swap on layers with large feature maps (e.g., conv1-4).

```
th test_wct.lua -content YourContentImagePath -style YourStyleImagePath -swap5 1
```
Below is an exemplary comparison between w/o and w/ swap operation on conv5. It is obsearved that with the swapping, the eyeball in the content is replaced with the ball in the style (bottom) as they are cloeset neighbours in whitened feature space.

<img src='figs/p1.jpg' width=840>


## Note

- In theory, the covariance matrix of whitened features should be Identity. In practise, it is not because we need to eliminate some extremely small eigen values (e.g., <1e-10) or add a small constant (e.g., 1e-7) to all eigen values in order to perform the inverse operation (D^-1/2) in the whitening.

- Our decoders trained for reconstruction is not perfect. As inverting deeper features (e.g., conv5_1) to RGB images is relatively difficult, we expect better decoders from researchers. If users prefer to preserve detailed structures in the content during the transferring, a more powerful decoder is necessary.

- To save memory for testing image of large size, we need to often load and delete model. So in our code, for the transferring on each content/style pair, we need to reload the model.

- We found that using "CUDA_VISIBLE_DEVICES=X" is better than using "-gpu X" as the former choice will guarantee that all weights/gradients/input will be located on the same GPU.

## Citation

```
@inproceedings{WCT-NIPS-2017,
    author = {Li, Yijun and Fang, Chen and Yang, Jimei and Wang, Zhaowen and Lu, Xin and Yang, Ming-Hsuan},
    title = {Universal Style Transfer via Feature Transforms},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2017}
}
```

## Acknowledgement

We express gratitudes to the great work [AdaIN](https://github.com/xunhuang1995/AdaIN-style) and [Style-swap](https://github.com/rtqichen/style-swap) as we benefit a lot from both their paper and codes.
