import os
import random

def save():
    trainval_percent = 1
    train_percent = 0.2
    imgfilepath = r''
    txtsavepath = r''
    total_xml = os.listdir(imgfilepath)

    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    # train.txt is a list of filenames of the image files used for training
    # val.txt is the list of filenames of the image files used for validation
    # trianval.txt is the list of filenames of the image files used for training and validation
    # test.txt is a list of filenames of the image files to be tested

    ftrain = open(txtsavepath + '/train.txt', 'w')
    ftrainval = open(txtsavepath + '/trainval.txt', 'w')
    # ftest = open(txtsavepath + '/test.txt', 'w')
    fval = open(txtsavepath + '/val.txt', 'w')
    print("begin!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        # else:
        #     ftest.write(name)
    print("Succeed!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    ftrainval.close()
    ftrain.close()
    fval.close()
    # ftest.close()
save()
