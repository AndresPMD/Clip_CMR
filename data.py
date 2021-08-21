import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import json


class GenericDataset(data.Dataset):
    """
    Dataset loader for all datasets.
    """

    def __init__(self, transform, split, data_name):
        self.transform = transform
        self.split = split
        self.data_name = data_name

        if self.data_name == 'coco':
            self.img_path = '/SSD/COCO_raw/val2014/'
        else:
            self.img_path = '/SSD/Datasets/Flickr30K/images/'
        
        with open ("./data/{}_images.txt".format(self.data_name), "r") as fp:
            self.images = fp.readlines()
        self.images = [image.strip() for image in self.images]
        
        with open ("./data/{}_captions.txt".format(self.data_name), "r") as fp:
            self.captions = fp.readlines()
        self.captions = [caption.strip() for caption in self.captions]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        image_name = self.images[index]
        caption = self.captions[index]
        caption = caption.split(' ')
        short_caption = caption[:40]
        # if index >= 3905:
        #     import pdb; pdb.set_trace()
        caption = ' '.join(short_caption)

        image = self.transform(Image.open(self.img_path + image_name))

        return image, caption, index, image_name

    def __len__(self):
        return len(self.images)

#################

def get_loader(transform, split, data_name, batch_size, num_workers, args,):
    """Returns torch.utils.data.DataLoader for custom dataset."""

    dataset = GenericDataset(transform, split, data_name)
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers)
    return data_loader

def get_test_loader(split, data_name, batch_size, workers, args, preprocess):

    # Build Dataset Loader
    transform = preprocess

    test_loader = get_loader(transform, args.split, args.data_name, args.batch_size, args.workers, args,)

    return test_loader

