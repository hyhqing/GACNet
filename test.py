import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
from get_model import get_model
from scipy.io import loadmat
from dataloader.dataset_4channels import readTif

parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str,default=r'D:\Desktop\test\\',
                    help='root dir for data')
parser.add_argument('--save_path', type=str, default=r'D:\Desktop\result\gray_global',
                    help='root dir for save prediction result')
parser.add_argument('--weights_path', type=str, default=r'E:\works\model_2\weights',
                    help='root dir for weights')
parser.add_argument('--num_classes', type=int,default=45,
                    help='Number of categories')
parser.add_argument('--img_size', type=int,default=256,
                    help='Image Size')
parser.add_argument('--use_aux', type=bool, default=False,
                    help='Whether to use an auxiliary loss function')
parser.add_argument('--data_channel', type=int, default=3,
                    help='Number of data set input channels')
parser.add_argument('--X_type', type=str,default='RGB',
                    help='X data type, such as "RGB", "GGLI", "GLI", "NGRDI", and so on')
parser.add_argument('--Y_type', type=str,default='DSM',
                    help='Y data type, such as "DSM"')
parser.add_argument('--model_name', type=str, default='MFIANet_b3',
                    help='model name of testing')
parser.add_argument('--head', type=bool, default=False,
                    help='Does the network require a segmentation head')
parser.add_argument('--head_type', type=str, default='seghead',
                    help='segmentation head name, such as "seghead", "fpnhead", "uperhead", "mlphead", and so on')
args = parser.parse_args()

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if args.data_channel == 3:
    model = get_model(args, models=args.model_name)
    model.to(DEVICE)
    model.load_state_dict(torch.load(args.weights_path))
    model.eval()

    test_root = os.path.join(args.test_path, 'images')
    X_path = os.path.join(test_root, args.X_type)
    Y_path = os.path.join(test_root, args.Y_type)
    img_names = os.listdir(X_path)
    for name in img_names:
        X_full_path = os.path.join(X_path, name)
        Y_full_path = os.path.join(Y_path, name)

        X_img = cv2.imread(X_full_path, cv2.IMREAD_UNCHANGED)
        if X_img.shape[2] == 4:
            X_img = X_img[:, :, :3]
        if len(X_img.shape) == 2:
            X_img = cv2.merge([X_img, X_img, X_img])
            
        X_image = np.array(X_img, np.float32) / 255.0
        X_image = np.array(X_image, np.float32).transpose(2, 0, 1)

        Y_img = cv2.imread(Y_full_path, cv2.IMREAD_UNCHANGED)
        Y_img = cv2.merge([Y_img, Y_img, Y_img])

        Y_image = np.array(Y_img, np.float32) / 255.0
        Y_image = np.array(Y_image, np.float32).transpose(2, 0, 1)

        X_image = np.expand_dims(X_image, axis=0)
        X_image = torch.Tensor(X_image)
        X_image = X_image.cuda()

        Y_image = np.expand_dims(Y_image, axis=0)
        Y_image = torch.Tensor(Y_image)
        Y_image = Y_image.cuda()

        with torch.no_grad():
            output = model(X_image, Y_image)
            output = F.softmax(output[0], dim=1)
            pred = torch.argmax(output, dim=1)
            pred = torch.squeeze(pred).cpu().numpy()

            img_name = name.split('.')[0] + '.png'
            save_full = os.path.join(args.save_path, img_name)
            cv2.imwrite(save_full, pred)

else:
    model = get_model(args, models=args.model_name)
    model.to(DEVICE)
    model.load_state_dict(torch.load(args.weights_path))
    model.eval()

    test_root = os.path.join(args.test_path, 'images')
    im_names = os.listdir(test_root)
    for name in im_names:
        full_path = os.path.join(test_root, name)
        img = readTif(full_path)
        image = np.array(img, np.float32) / 255

        image = np.expand_dims(image, axis=0)
        image = torch.Tensor(image)
        image = image.cuda()
        with torch.no_grad():
            output = model(image)
            if isinstance(output, (list, tuple)):
                output = F.softmax(output[0], dim=1)
            else:
                output = F.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1)
            pred = torch.squeeze(pred).cpu().numpy()

            img_name = name.split('.')[0] + '.png'
            save_full = os.path.join(args.save_path, img_name)
            cv2.imwrite(save_full, pred)