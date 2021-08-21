import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import json


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, vocab, transform=None, ids=None):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
        """
        self.root = root
        # when using `restval`, two json files are needed
        if isinstance(json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)
        # if ids provided by get_paths, use split-specific ids
        if ids is None:
            self.ids = list(self.coco.anns.keys())
        else:
            self.ids = ids

        # if `restval` data is to be used, record the break point for ids
        if isinstance(self.ids, tuple):
            self.bp = len(self.ids[0])
            self.ids = list(self.ids[0]) + list(self.ids[1])
        else:
            self.bp = len(self.ids)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root, caption, img_id, path, image = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def get_raw_item(self, index):
        if index < self.bp:
            coco = self.coco[0]
            root = self.root[0]
        else:
            coco = self.coco[1]
            root = self.root[1]
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(root, path)).convert('RGB')

        return root, caption, img_id, path, image

    def __len__(self):
        return len(self.ids)


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
        image = self.images[index]
        caption = self.captions[index]

        
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

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

