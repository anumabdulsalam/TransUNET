import os
import random
import h5py
import numpy as np
import torch
import cv2
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


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


class Queens_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, nclass=9, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.nclass = nclass

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            #print(image.shape)
            #image = np.reshape(image, (512, 512))
            #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            #label = np.reshape(label, (512, 512))
            
        elif self.split=="val_vol":
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
            #image = np.reshape(image, (image.shape[2], 512, 512))
            #label = np.reshape(label, (label.shape[2], 512, 512))
            #label[label==5]= 0
            #label[label==9]= 0
            #label[label==10]= 0
            #label[label==12]= 0
            #label[label==13]= 0
            #label[label==11]= 5

        if self.nclass == 9:
            label[label==5]= 0
            label[label==9]= 0
            label[label==10]= 0
            label[label==12]= 0
            label[label==13]= 0
            label[label==11]= 5
            
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
if __name__=='__main__':
    from torchvision import transforms
    dataset=Queens_dataset(base_dir='../data/NIDI/train_npz_new/', list_dir='../lists/lists_NIDI', split="train", nclass=9,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[256, 256])]))
    db_test = Queens_dataset(base_dir='../data/NIDI/test_vol_h5_new/', split="test_vol", list_dir='../lists/lists_NIDI',nclass=9)
    print(dataset[0]['image'].shape)
    print(db_test[0]['image'].shape)