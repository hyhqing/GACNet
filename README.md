# ⭐️⭐️MFIANet Research Code Help 

## Catalog

### 		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. Introduction

### 		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. File Directory

### 		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. Performance

### 		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. Necessary Environments

### 		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5. Data Preparation

### 		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6. Train , Test and Evaluate

### 		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7. Download Link

### 		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;8. Citation

## Introduction

#### This is the official code of MFIANet: A Multilevel Feature Interaction and Aggregation-Based Network for Extraction of Citrus Tree Canopy Height under Complex Terrain. a Multilevel Feature Interaction and Aggregation Network (MFIANet) for estimating citrus tree heights from unmanned aerial vehicle photogrammetry-derived data (UAVPD). In the encoding stage of the model, we designed a Multisource Feature Interaction Module (MFIM) and a Multisource Feature Aggregation Module (MFAM). These modules recalibrate feature responses from diverse datasets through triple feature interactions involving position, channel, and spatial correlation, and alternately aggregate the recalibrated features. In the decoding stage, to significantly reduce computational costs, we designed a Multilevel Feature Aggregation Decoder (MLFA-Decoder) based on cross-attention to ensure context consistency and enable the efficient extraction of citrus tree canopy and height. We evaluated our method in four citrus tree growing areas with different topographic conditions.



## File Directory



| Filename     | Brief                                                        |
| ------------ | ------------------------------------------------------------ |
| train.py     | The model training startup file, and configuration about the model |
| test.py      | The model testing startup file, and configuration about the parameter |
| savetxt.py   | Code to generate txt files of training, validation and test data |
| trainer.py   | Hyperparameter profile for model training process            |
| get_model.py | Model Load File                                              |



## Performance

#### Performance on tree canopy segmentation and tree height estimation between the MFIANet model and state-of-the-art Networks (using RGB-DSM data as input)



|      Method      |     Modal     |   OA(%)   |   F1(%)   |  mIoU(%)  | MAE (m)  | RMSE(m)  |
| :--------------: | :-----------: | :-------: | :-------: | :-------: | :------: | :------: |
|       FCN        | Channel Stack |   94.95   |   90.02   |   87.76   |   0.44   |   0.70   |
|       UNet       | Channel Stack |   95.52   |   92.05   |   90.56   |   0.36   |   0.62   |
|    HRCNet_W48    | Channel Stack |   96.25   |   93.74   |   92.61   |   0.34   |   0.56   |
|    DeepLabV3+    | Channel Stack |   96.36   |   93.55   |   91.96   |   0.30   |   0.51   |
|  EfficientNetV2  | Channel Stack |   95.67   |   92.44   |   90.50   |   0.29   |   0.48   |
|    TransUNet     | Channel Stack |   96.62   |   93.64   |   91.91   |   0.30   |   0.50   |
|    CSwin-Tiny    | Channel Stack |   96.71   |   93.31   |   91.60   |   0.33   |   0.54   |
|   CSwin-Small    | Channel Stack |   96.92   |   93.81   |   91.80   |   0.32   |   0.51   |
|    CSwin-Base    | Channel Stack |   97.01   |   93.99   |   92.02   |   0.30   |   0.49   |
|   CSwin-Large    | Channel Stack |   96.86   |   93.80   |   91.76   |   0.31   |   0.51   |
|   SegFormer_B2   | Channel Stack |   96.83   |   93.01   |   91.61   |   0.33   |   0.52   |
|   SegFormer_B3   | Channel Stack |   96.90   |   93.33   |   91.90   |   0.31   |   0.50   |
|   SegFormer_B4   | Channel Stack |   96.95   |   93.81   |   92.29   |   0.29   |   0.47   |
| MFIANet (MiT-b2) |    RGB-DSM    |   97.94   |   96.03   |   94.86   |   0.27   |   0.42   |
| MFIANet (MiT-b3) |    RGB-DSM    | **98.02** | **96.19** | **95.08** | **0.25** | **0.38** |
| MFIANet (MiT-b4) |    RGB-DSM    |   97.87   |   95.91   |   94.69   |   0.26   |   0.42   |

## Necessary Environments

### Install

1. ```
   git clone https://github.com/xxxxZhou/MFIANet $MFIANet_ROOT
   ```

2. If you want to train and test our models on your datasets, you need to install dependencies: pip install -r requirements.txt

## Data Preparation

#### If you want to train or test your data on our model, you need to prepare the data in the following format.

#### Your directory tree should be look like this:

```python
MFIANet_ROOT/data
├── train_val
│   ├── images
│   │   ├── DSM
│   │   ├── RGB
│   │   ├── GGLI
│   │   ├── GLI
│   │   └── NGRDI
│   └── labels
├── test
│   ├── images
│   │   ├── DSM
│   │   ├── RGB
│   │   ├── GGLI
│   │   ├── GLI
│   │   └── NGRDI
│   └── labels
├── list
│   ├── train_val
│   │   ├── train.txt
│   │   ├── val.txt
│   │   └── trainval.txt
```

## Train , Test and Evaluate

### Training

#### Just specify the configuration file for train.py. For example, configure the following parameters in train.py or enter the command line directly in the terminal.

#### Command Lines:

```
train.py --root_path F:\images_dataset --list_dir E:\works\list --num_classes 45 --img_size 256 --batch_size 8 --max_epochs 50 --output_dir E:\works --init_lr 0.0001 --data_channel 3 --X_type RGB --Y_type DSM --model_name MFIANet_b2 --head False --train_para_file MFIANet_b2_train_para --val_para_file MFIANet_b2_val_para
```

### Testing

### MFIANet_b2

#### Configure the following parameters in test.py or enter the command line directly in the terminal.

```
test.py --test_path D:\Desktop\test\ --save_path D:\Desktop\result\ --weights_path E:\works\model_2\weights --num_classes 45 --img_size 256 --data_channel 3 --X_type RGB --Y_type DSM --model_name MFIANet_b2
```

![图12](https://raw.gitmirror.com/xxxxZhou/MFIANet/main/undefined202406271423927.png)

### Evaluating

#### To evaluate the accuracy of the prediction results, we use the evaluate index.py code alone.

## Download Link

#### Pre-trained model weights files, link:

##### segformer_b0: https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b0_backbone_imagenet-eb42d485.pth,

##### segformer_b1: https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b1_backbone_imagenet-357971ac.pth,

##### segformer_b2: https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b2_backbone_imagenet-3c162bb8.pth,

##### segformer_b3: https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b3_backbone_imagenet-0d113e32.pth,

##### segformer_b4: https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b4_backbone_imagenet-b757a54d.pth,

##### segformer_b5: https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b5_backbone_imagenet-d552b33d.pth

## Citation

#### If you find this work or code is helpful in your research, please cite:

```
@inproceedings{HeZZCW24,
  title={MFIANet: A Multilevel Feature Interaction and Aggregation-Based Network for Extraction of Citrus Tree Canopy Height under Complex Terrain},
  author={Haiqing He and Fuyang Zhou and Yongjun Zhang and Ting Chen and Yan Wei},
  booktitle={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024}
}
```
