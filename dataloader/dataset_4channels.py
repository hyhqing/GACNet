# -*- coding: utf-8 -*-
import os
import cv2
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from osgeo import gdal


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask


def read_data(root_path, list_dir, mode = 'train'):
    sample_list = open(os.path.join(list_dir, mode + '.txt')).readlines()
    images = []
    masks = []
    image_root = os.path.join(root_path, 'images')
    gt_root = os.path.join(root_path, 'labels')

    for i in range(len(sample_list)):
        slice_name = sample_list[i].strip('\n')
        image_name = slice_name + '.png'
        image_path = os.path.join(image_root, image_name)

        label_name = slice_name + '.png'
        label_path = os.path.join(gt_root, label_name)

        images.append(image_path)
        masks.append(label_path)

    return images, masks


def readTif(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    band = dataset.RasterCount
    GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
    inf = np.isinf(GdalImg_data)
    GdalImg_data[inf] = 255
    nan = np.isnan(GdalImg_data)
    GdalImg_data[nan] = 255
    return GdalImg_data

#训练数据读取
def read_val_data(img_path, mask_path):
    img = readTif(img_path)
    mask = cv2.imread(mask_path, 0)

    mask = np.expand_dims(mask, axis=2)

    img = np.array(img, np.float32) / 255.0
    mask = np.array(mask, np.float32)

    img = np.array(img, np.float32)
    mask = np.array(mask, np.float32).transpose(2, 0, 1)

    return img, mask

#验证数据读取
def read_train_data(img_path, mask_path):
    img = readTif(img_path)
    mask = cv2.imread(mask_path, 0)
    mask = np.expand_dims(mask, axis=2)

    img = np.array(img, np.float32) / 255.0
    mask = np.array(mask, np.float32)

    img = np.array(img, np.float32)
    mask = np.array(mask, np.float32).transpose(2, 0, 1)

    return img, mask

class ImageFolder(Dataset):
    def __init__(self, root_path, list_dir, mode='train'):
        self.root = root_path
        self.mode = mode
        self.list_dir = list_dir
        self.images, self.labels = read_data(self.root, self.list_dir, self.mode)

    def __getitem__(self, index):
        if self.mode == 'val':
            img, mask = read_val_data(self.images[index], self.labels[index])
            img = torch.Tensor(img)
            mask = torch.Tensor(mask)
        else:
            img, mask = read_train_data(self.images[index], self.labels[index])
            img = torch.Tensor(img)
            mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        # assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)