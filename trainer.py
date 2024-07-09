import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from tool.loss import DiceLoss, BinaryDiceLoss
from pytorch_toolbelt import losses as L
from datetime import datetime


np.seterr(divide='ignore',invalid='ignore')


def worker_init_fn():
    random.seed(0)

def poly_adjust_learning_rate(optimizer, base_lr, max_iters,
        cur_iters, warmup_iter=None, power=0.9):
    if warmup_iter is not None and cur_iters < warmup_iter:
        lr = base_lr * cur_iters / (warmup_iter + 1e-8)
    elif warmup_iter is not None:
        lr = base_lr*((1-float(cur_iters - warmup_iter) / (max_iters - warmup_iter))**(power))
    else:
        lr = base_lr * ((1 - float(cur_iters / max_iters)) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    return lr

def calculate_accuracy(outputs, targets):
    outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
    outputs = torch.transpose(outputs, 1, 2).contiguous()
    outputs = outputs.view(-1, outputs.size(2))
    targets = targets.view(-1)

    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy

def trainer_synapse(args, model, save_path):
    if args.data_channel == 3:
        from dataloader.dataset_3channels import ImageFolder
    elif args.data_channel == 4:
        from dataloader.dataset_4channels import ImageFolder

    logging.basicConfig(filename=save_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    model_name = args.model_name
    base_lr = args.init_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    warm_epoch = args.warm_epoch

    db_train = ImageFolder(args.root_path, args.list_dir, args.X_type, args.Y_type, mode='train')
    db_val = ImageFolder(args.root_path, args.list_dir, args.X_type, args.Y_type, mode='val')

    print("\033[1;33;44m The length of train set is: %d \033[0m" % (len(db_train)))
    print("\033[1;33;44m The length of val set is: %d \033[0m" % (len(db_val)))

    train_loader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn)

    val_loader = DataLoader(
        db_val,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    bce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-3)

    iter_num = 0
    train_time = []
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)
    logging.info("\033[1;33;44m %d iterations per epoch. %d max iterations.\033[0m" % (len(train_loader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        import time
        start = time.time()
        train_loss = 0
        val_loss = 0
        train_accuracy = 0
        val_accuracy = 0

        model.train()
        for iteration, batch in enumerate(train_loader):
            losses = 0
            if iteration >= len(train_loader):
                break
            rgb_imgs, x_imgs, pngs= batch
            rgb_imgs, x_imgs, pngs= rgb_imgs.cuda(), x_imgs.cuda(), pngs.cuda()
            optimizer.zero_grad()
            outputs = model(rgb_imgs, x_imgs)

            if isinstance(outputs, (list, tuple)):
                for i in range(len(outputs)):
                    output = outputs[i]
                    pngs = torch.squeeze(pngs)
                    pngs = pngs.long()
                    if i > 0:
                        loss = 0.5 * bce_loss(output, pngs)
                    else:
                        loss = bce_loss(output, pngs)
                    losses += loss
            else:
                output = outputs
                pngs = torch.squeeze(pngs)
                pngs = pngs.long()
                losses = bce_loss(output, pngs)

            losses = losses.mean()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

            accuracy = calculate_accuracy(output, pngs)
            train_accuracy += accuracy

            iter_num += 1
            lr = poly_adjust_learning_rate(optimizer, base_lr, max_iterations, iteration + epoch_num * len(train_loader),
                                      warm_epoch * len(train_loader))

            logging.info('\033[32;1m iteration %d | train_loss: %f | train_accuracy: %f | lr: %f \033[0m' % (iter_num, losses.item(), accuracy, lr))

            table_iteration = {"iteration":iteration + 1,
                               "train_loss":train_loss / (iteration + 1),
                               "train_accuracy":train_accuracy / (iteration + 1)}
            data_iteration = pd.DataFrame(table_iteration, index=[[0]])
            if not os.path.exists(args.output_dir + '/' + args.train_para_file + '_iteration' + '.csv'):
                data_iteration.to_csv(args.output_dir + '/' + args.train_para_file + '_iteration' + '.csv', mode='a',
                                      header=True, index=False)
            else:
                data_iteration.to_csv(args.output_dir + '/' + args.train_para_file + '_iteration' + '.csv', mode='a',
                                      header=False, index=False)

        times = "%s" % datetime.now()
        table = {"time":times,
                 "epoch":epoch_num,
                 "train_loss":train_loss / len(train_loader),
                 "train_accuracy":train_accuracy / len(train_loader)}
        data = pd.DataFrame(table, index=[[0]])
        if not os.path.exists(args.output_dir + '/' + args.train_para_file + '.csv'):
            data.to_csv(args.output_dir + '/' + args.train_para_file + '.csv', mode='a',
                                  header=True, index=False)
        else:
            data.to_csv(args.output_dir + '/' + args.train_para_file + '.csv', mode='a',
                                  header=False, index=False)

        model.eval()
        for iteration, batch in enumerate(val_loader):
            losses = 0
            if iteration >= len(val_loader):
                break
            rgb_imgs, x_imgs, pngs= batch
            rgb_imgs, x_imgs, pngs= rgb_imgs.cuda(), x_imgs.cuda(), pngs.cuda()
            with torch.no_grad():
                outputs = model(rgb_imgs, x_imgs)

                if isinstance(outputs, (list, tuple)):
                    for i in range(len(outputs)):
                        output = outputs[i]
                        pngs = torch.squeeze(pngs)
                        pngs = pngs.long()
                        if i > 0:
                            loss = 0.5 * bce_loss(output, pngs)
                        else:
                            loss = bce_loss(output, pngs)
                        losses += loss
                else:
                    output = outputs
                    pngs = torch.squeeze(pngs)
                    pngs = pngs.long()
                    losses = bce_loss(output, pngs)

                losses = losses.mean()
                val_loss += losses.item()

                accuracy = calculate_accuracy(output, pngs)
                val_accuracy += accuracy

                table_iteration = {"iteration": iteration + 1,
                                   "train_loss": train_loss / (iteration + 1),
                                   "train_accuracy": train_accuracy / (iteration + 1)}
                data_iteration = pd.DataFrame(table_iteration, index=[[0]])
                if not os.path.exists(args.output_dir + '/' + args.val_para_file + '_iteration' + '.csv'):
                    data_iteration.to_csv(args.output_dir + '/' + args.val_para_file + '_iteration' + '.csv', mode='a',
                                          header=True, index=False)
                else:
                    data_iteration.to_csv(args.output_dir + '/' + args.val_para_file + '_iteration' + '.csv', mode='a',
                                          header=False, index=False)
        end = time.time()
        time = end - start
        train_time.append(time)

        print("\n\033[1;33;44m The %s epoch training loss is %f, validation loss is %f, training accuracy is %f, "
              "validation accuracy is %f,  training and validation time are %fs, ""Total training time is %fs \033[0m]" % (
            epoch_num, train_loss/len(train_loader), val_loss/len(val_loader), train_accuracy/len(train_loader),
            val_accuracy/len(val_loader),time, sum(train_time)))

        times = "%s" % datetime.now()  # 获取当前时间
        table = {"time": times,
                 "epoch": epoch_num,
                 "val_loss": val_loss / len(val_loader),
                 "val_accuracy":val_accuracy / len(val_loader),
                 "tran_time": sum(train_time)}
        data = pd.DataFrame(table, index=[[0]])
        if not os.path.exists(args.output_dir + '/' + args.val_para_file + '.csv'):
            data.to_csv(args.output_dir + '/' + args.val_para_file + '.csv', mode='a',
                                  header=True, index=False)
        else:
            data.to_csv(args.output_dir + '/' + args.val_para_file + '.csv', mode='a',
                                  header=False, index=False)

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(save_path, model_name + '_epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    return "Training Finished!"