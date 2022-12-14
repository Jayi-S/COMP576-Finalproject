Comp576 final Project:  clothing segmatation

Teammates: Jiayi Song & Hongtao Yang

## Installation

- Numpy `pip install numpy`
- Pillow `pip install pillow`
- OpenCV `pip install opencv-python`
- Tensorflow `pip install tensorflow-gpu=1.14`  
- scikit-learn `pip install scikit-learn`

## Dataset Processing

original dataset url: https://www.kaggle.com/competitions/imaterialist-fashion-2019-FGVC6/data

We have finished processing original data, each image is resized to 256 x 256. 

The dataset has the following structure.

```buildoutcfg
|-- clothe_seg
|  |-- train
|  |  |-- images
|  |  |-- labels
|  |  |-- clabels
|  |-- valid
|  |  |-- images
|  |  |-- labels
|  |  |-- clabels
|  |-- test
|  |  |-- images
|  |  |-- labels
|  |  |-- clabels
|  |-- class_dict.csv
|  |-- evaluated_classes

```
Folder `images` has clothe images

Folder  `labels` has  clothe samples.

Folder `clabels` contains color label images. It' s not used to train or test, it's just for visualization.

We put part of dataset under this project.

### Training

You can directly run the following command to start training

```buildoutcfg
python train.py --model UNet --base_model VGG16 --dataset "clothe_seg" --num_classes 47 --loss 'miou_loss' --batch_size 16 --learning_rate 1e-3
```
To use different loss functions， change the value of parameter`--loss`,you can use

`ce`

`dice_loss`

`dice_ce_loss`

`focal_loss`

`miou_loss`

### Predicting

We provide one trained model weights. You can use it to get the prediction of a single RGB image:
```buildoutcfg
python predict.py --model UNet --base_model VGG16 --num_classes 47 --weights checkpoints/UNet_based_on_VGG16_miou_loss.h5 --image_path "clothe_seg/test/images" --color_encode True
```

The predicting results will be under folder `predictions`

If you have other trained models, you can ag

### Testing

To do testing, you should firstly doing prediction to get the mask. For example,  if you use our pretrained checkpoint. Firstly run

```buildoutcfg
python predict.py --model UNet --base_model VGG16 --num_classes 47 --weights checkpoints/UNet_based_on_VGG16_miou_loss.h5 --image_path "clothe_seg/test/images"  --csv_file "clothe_seg/class_dict.csv" --color_encode False
```

It will output gray-scale images under folder `predictions`.

Then use `evaluate.py` to evaluate the performance.

```
python evaluate.py --dataset 'clothe_seg' --predictions predictions
```

