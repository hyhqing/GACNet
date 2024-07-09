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
# from osgeo import gdal
from PIL import Image


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

def read_data(root_path, list_dir, X_type, Y_type, mode = 'train'):
    sample_list = open(os.path.join(list_dir, mode + '.txt')).readlines()
    X_images = []
    Y_images = []
    masks = []
    X_image_root = os.path.join(root_path, 'images')
    X_image_root = os.path.join(X_image_root , X_type)

    Y_image_root = os.path.join(root_path, 'images')
    Y_image_root = os.path.join(Y_image_root, Y_type)

    gt_root = os.path.join(root_path, 'labels')

    for i in range(len(sample_list)):
        slice_name = sample_list[i].strip('\n')
        image_name = slice_name + '.png'
        X_image_path = os.path.join(X_image_root, image_name)
        Y_image_path = os.path.join(Y_image_root, image_name)

        label_name = slice_name + '.png'
        label_path = os.path.join(gt_root, label_name)

        X_images.append(X_image_path)
        Y_images.append(Y_image_path)
        masks.append(label_path)

    return X_images, Y_images, masks

def read_train_data(X_img_path, Y_img_path, mask_path):
    X_img = cv2.imread(X_img_path, cv2.IMREAD_UNCHANGED)
    if len(X_img.shape) == 2:
        X_img = cv2.merge([X_img, X_img, X_img])
    if X_img.shape[2] == 4:
        X_img = X_img[:, :, :3]

    Y_img = cv2.imread(Y_img_path, cv2.IMREAD_UNCHANGED)
    Y_img = cv2.merge([Y_img, Y_img, Y_img])

    mask = cv2.imread(mask_path, 0)
    mask = np.expand_dims(mask, axis=2)

    X_img = np.array(X_img, np.float32) / 255.0
    Y_img = np.array(Y_img, np.float32) / 255.0
    mask = np.array(mask, np.float32)

    X_img = np.array(X_img, np.float32).transpose(2, 0, 1)
    Y_img = np.array(Y_img, np.float32).transpose(2, 0, 1)
    mask = np.array(mask, np.float32).transpose(2, 0, 1)

    return X_img, Y_img, mask


def read_val_data(X_img_path, Y_img_path, mask_path):
    X_img = cv2.imread(X_img_path, cv2.IMREAD_UNCHANGED)
    if len(X_img.shape) == 2:
        X_img = cv2.merge([X_img, X_img, X_img])
    if X_img.shape[2] == 4:
        X_img = X_img[:, :, :3]

    Y_img = cv2.imread(Y_img_path, cv2.IMREAD_UNCHANGED)
    Y_img = cv2.merge([Y_img, Y_img, Y_img])

    mask = cv2.imread(mask_path, 0)
    mask = np.expand_dims(mask, axis=2)

    X_img = np.array(X_img, np.float32) / 255.0
    Y_img = np.array(Y_img, np.float32) / 255.0

    mask = np.array(mask, np.float32)

    X_img = np.array(X_img, np.float32).transpose(2, 0, 1)
    Y_img = np.array(Y_img, np.float32).transpose(2, 0, 1)
    mask = np.array(mask, np.float32).transpose(2, 0, 1)

    return X_img, Y_img, mask

class ImageFolder(Dataset):
    def __init__(self, root_path, list_dir, X_type, Y_type, mode='train'):
        self.root = root_path
        self.mode = mode
        self.list_dir = list_dir
        self.x_type = X_type
        self.y_type = Y_type
        self.X_images, self.Y_images, self.labels = read_data(self.root, self.list_dir, self.x_type, self.y_type, self.mode)

    def __getitem__(self, index):
        if self.mode == 'val':
            X_img, Y_img, mask = read_val_data(self.X_images[index], self.Y_images[index], self.labels[index])
            X_img = torch.Tensor(X_img)
            Y_img = torch.Tensor(Y_img)
            mask = torch.Tensor(mask)

        else:
            X_img, Y_img, mask= read_train_data(self.X_images[index], self.Y_images[index], self.labels[index])
            X_img = torch.Tensor(X_img)
            Y_img = torch.Tensor(Y_img)
            mask = torch.Tensor(mask)

        return X_img, Y_img, mask

    def __len__(self):
        # assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.X_images)