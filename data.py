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
            self.images = readlines(fp)
        
        with open ("./data/{}_captions.txt".format(self.data_name), "r") as fp:
            self.captions = readlines(fp)

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        image_name = self.images[index]
        caption = self.captions[index]

        image = self.transform(Image.open(self.img_path + image_name)).unsqueeze(0)

        return image, caption, index, image_name

    def __len__(self):
        return len(self.ids)


#################

def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids, caption_labels, caption_masks, categories = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    caption_labels_ = torch.stack(caption_labels, 0)
    caption_masks_ = torch.stack(caption_masks, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    categories = torch.stack(categories, 0)

    return images, targets, lengths, ids, caption_labels_, caption_masks_, categories


def get_loader(transform, split, data_name, batch_size, num_workers, args, collate_fn=collate_fn):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

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

        
        
        opt.data_name, split_name,
                                    roots[split_name]['img'],
                                    roots[split_name]['cap'],
                                    vocab, transform, ids=ids[split_name],
                                    batch_size=batch_size, shuffle=False,
                                    num_workers=workers,
                                    collate_fn=collate_fn)

    return test_loader

