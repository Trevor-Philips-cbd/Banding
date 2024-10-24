## Dependencies
* Python 3.8
* PyTorch >= 1.7.1
* numpy
* skimage
* **imageio**
* matplotlib
* tqdm

## Dataset
We use DIV2K dataset as clear images to train our model. Please download it from <a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">here</a>  or  <a href="https://cv.snu.ac.kr/research/EDSR/DIV2K.tar">SNU_CVLab</a>. Put all clear images into the dataset/DIV2K/DIV2K_train_HR.
As for noisy images, we use Matlab/generate_noise.m to generate noisy images and put them into the dataset/DIV2K/DIV2K_train_LR_bicubic/x1.

When testing, you can put the clear images and noisy images of the test set into dataset/DIV2K/DIV2K_train_HR and dataset/DIV2K/DIV2K_train_LR_bicubic/x1 respectively

##Training

Using --ext sep_reset argument on your first running. 

You can skip the decoding part and use saved binaries with --ext sep argument in second time.

```python
## train
python main.py --scale 1 --patch_size 176 --save ewt --ext sep_reset
```

##Testing
All pre-trained model should be put into experiment/ first.
```python
## test
python main.py --data_test DIV2K --data_range 1-24 --scale 1 --pre_train your_path/EWT/experiment/model_name/model/model_best.pt --test_only --save_results --ext sep_reset
```
After the above command is run, a file named test will be generated in experiment/, where you can view the noise-removed image.
