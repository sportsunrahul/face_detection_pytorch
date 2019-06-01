import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import random_split

import cv2
import numpy as np
import glob



class FaceDataset(Dataset):

    def __init__(self, targets, file_names, transform=None):
        self.targets = targets
        self.file_names = file_names
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        image = cv2.imread(img_name)
        targets = self.targets[idx]

        if self.transform:
            X = self.transform(image)
        
        return (X,targets)

def load_dataset(opt):
    train_num = 10000
    train_face_file = glob.glob(opt.train_face)
    train_nonface_file = glob.glob(opt.train_nonface)
    train_file_names = train_face_file[:train_num] + train_nonface_file[:train_num]
    targets = np.append(np.ones((train_num)), np.zeros((train_num)))




    test_face_file = glob.glob(opt.test_face)
    test_nonface_file = glob.glob(opt.test_nonface)
    test_file_names = test_face_file[:1000] + test_nonface_file[:1000]
    test_targets = np.append(np.ones((1000)), np.zeros((1000)))

    data = []
    for i in train_file_names:
        data.append(cv2.imread(i))
    data = np.array(data)
    mean = data.mean(axis=(0,1,2))/255
    std = data.std(axis=(0,1,2))/255
    # print(mean, std)
    data = []



    train_dataset = FaceDataset(targets, train_file_names, transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean), (std))]))
    test_set = FaceDataset(test_targets, test_file_names, transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean), (std))]))

    # print(train_dataset.__len__())
    train_len = int(opt.train_val_split * train_dataset.__len__())
    val_len = train_dataset.__len__() - train_len
    train_set, val_set = random_split(train_dataset, (train_len, val_len))

    print("Train Samples = {}".format(train_set.__len__()))
    print("Validation Samples = {}".format(val_set.__len__()))


    return train_set, val_set, test_set

if __name__ == "__main__":
    from opts import opts
    opt = opts().parse()
    load_dataset(opt)
    
