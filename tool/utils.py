import numpy as np

def label_to_RGB(image, classes=2):
    RGB = np.zeros(shape=[image.shape[0], image.shape[1], 3], dtype=np.uint8)
    if classes == 2:  # potsdam and vaihingen
        # palette = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
        palette = [[255, 255, 255], [0, 0, 255]]
    if classes == 4:  # barley
        palette = [[255, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    for i in range(classes):
        index = image == i
        RGB[index] = np.array(palette[i])
    return RGB

def deeplab_dataset_collate(batch):
    images, pngs, seg_labels = [], [], []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)

    # images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    # pngs = torch.from_numpy(np.array(pngs)).long()
    # seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    images = np.array(images)
    pngs = np.array(pngs)
    seg_labels = np.array(seg_labels)

    return images, pngs, seg_labels