# UniversalStyleTransfer
-------------------------------------------------------------------
WCT code
-------------------------------------------------------------------

The code is for style transfer only

For a single pair test:

   th XXX.lua -content YourContentImagePath -style YourStyleImagePath -alpha YourStyleWeight

For large numbers of pair test:

   th XXX.lua -contentDir YourContentImageDir -styleDir YourStyleImageDir -alpha YourStyleWeight

By default, we perform WCT (whitening and coloring transform) on relu1/2/3/4/5_1 features. 

We provide a parameter "-swap5 1" to perform swap operation on relu5_1 features. 


---------------------------------------------------------------------

1> test_wct.lua

   Both GPU and CPU are supported. Usage:

   for GPU: CUDA_VISIBLE_DEVICES=X th XXX.lua 
   for CPU: th XXX.lua -gpu -1

*** I found that using "CUDA_VISIBLE_DEVICES=X" is better than using "-gpu X" as the former choice will guarantee that all weights/gradients/input will be located on the same GPU.

2> test_wct_fater_gpu_only.lua

   Only GPU is supported. Usage:
   CUDA_VISIBLE_DEVICES=X th XXX.lua 

   For a GPU of memory ~12G, it is suggested that the contentSize and styleSize are all less than 900 (800 recommended for the largest size).

3>  test_wct_spatial2style.lua

   Spatial control: Style1 for foreground (mask==1), Style2 for background (mask==0), provided a binary mask

   Both GPU and CPU are supported. Usage:
   for GPU: CUDA_VISIBLE_DEVICES=X th test_wct_spatial2style.lua -content YourConentPath -style YourStylePath1,YourStylePath2 -mask YourBinaryMaskPath

   for CPU: th test_wct_spatial2style.lua -content YourConentPath -style YourStylePath1,YourStylePath2 -mask YourBinaryMaskPath -gpu -1

----------------------------------------------------------------------

Note:

To save memory for testing image of large size, we need to often load and delete model. So in our code, for the transferring on each content/style pair, we need to reload the model.
