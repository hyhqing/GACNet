import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from trainer import trainer_synapse
from get_model import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,default=r'F:\images_dataset',
                    help='Path to image and label data')
parser.add_argument('--list_dir', type=str,default=r'E:\works',
                    help='Specify txt file paths for training and validation data images')
parser.add_argument('--num_classes', type=int,default=45,
                    help='Number of categories')
parser.add_argument('--img_size', type=int,default=256,
                    help='Image Size')
parser.add_argument('--batch_size', type=int,default=8,
                    help='batch size')
parser.add_argument('--max_epochs', type=int,default=50,
                    help='maximum epoch number to train')
parser.add_argument('--warm_epoch', type=int, default=0,
                    help='warm epoch')
parser.add_argument('--n_gpu', type=int, default=1,
                    help='total gpu')
parser.add_argument('--output_dir', type=str, default=r'E:\works',
                    help='Path to weights and training parameter file output')
parser.add_argument('--deterministic', type=bool,  default=False,
                    help='whether use deterministic training')
parser.add_argument('--init_lr', type=float,  default=0.0001,
                    help='Initial learning rate for network training')
parser.add_argument('--use_aux', type=bool, default=False,
                    help='Whether to use an auxiliary loss function')
parser.add_argument('--data_channel', type=int, default=3,
                    help='Number of data set input channels')
parser.add_argument('--seed', type=int,default=0,
                    help='random seed')
parser.add_argument('--X_type', type=str,default='RGB',
                    help='X data type, such as "RGB", "GGLI", "GLI", "NGRDI", and so on')
parser.add_argument('--Y_type', type=str,default='CHM',
                    help='Y data type, such as "DSM"')
parser.add_argument('--model_name', type=str, default='GACNet_b2',
                    help="model for training")
parser.add_argument('--head', type=bool, default=False,
                    help='Does the network require a segmentation head')
parser.add_argument('--head_type', type=str, default='seghead',
                    help='segmentation head name, such as "seghead", "fpnhead", "uperhead", "mlphead", and so on')
parser.add_argument('--train_para_file', type=str, default='GACNet_b2_train_para',
                    help='Save the file of training parameters')
parser.add_argument('--val_para_file', type=str, default='GACNet_b2_val_para',
                    help='Save the file of validation parameters')
args = parser.parse_args()

if __name__ == "__main__":

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda')

    if not args.deterministic:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    net = get_model(args, models=args.model_name)

    trainer_synapse(args, net, args.output_dir)
