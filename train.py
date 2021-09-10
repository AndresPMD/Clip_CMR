import pickle
import os
import time
import shutil

import os
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
import argparse
from model import *
import data

from evaluation_models import *

import logging
import clip

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='wiki',
                        help="{coco, f30k, wiki}")
    parser.add_argument('--cnn', default='ViT-B/16',
                        help='[RN50, RN101, RN50x4, RN50x16, ViT-B/32, ViT-B/16]')
    # TRAINING PARAMS
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')


    print ("Linear Probe Trainer of CLIP on Image-Text Matching Datasets \n")
    args = parser.parse_args()
    print ("Arguments: ", args)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    # load model and options
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print ("Running on: ", device)

    model_clip, preprocess = clip.load(args.cnn, device=device)

    for param in model_clip.parameters():
        param.requires_grad = False

    model = Clip_Linear(model_clip, args)
    model.cuda()
    # Loss and Optimizer
    criterion = ContrastiveLoss(margin=args.margin, measure=args.measure, max_violation=args.max_violation)
    criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Get data loaders
    train_loader = data.get_split_loader('train', args.data_name, args.batch_size, args.workers, args, preprocess)
    val_loader = data.get_split_loader('val', args.data_name, args.batch_size, args.workers, args, preprocess)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            torch.load(args.resume)
            validate(args, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Train the Model
    best_rsum = 0
    for epoch in range(args.num_epochs):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train(args, train_loader, model, epoch, val_loader, criterion, optimizer)

        # evaluate on validation set
        rsum = validate(args, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'args': args,
        }, is_best, prefix=args.logger_name + '/')



def train(args, train_loader, model, epoch, val_loader, criterion, optimizer):
    # average meters to record the training statistics


    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, captions, index, image_names) in enumerate(train_loader):
        # Always reset to train mode, this is not the default behavior
        model.train()

        captions = torch.cat([clip.tokenize(c) for c in captions])
        # compute the embeddings
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Update the model
        img_emb, cap_emb = model(images, captions)
        optimizer.zero_grad()
        loss = criterion(img_emb, cap_emb)
        loss.backward()

        # if args.grad_clip > 0:
        #     clip_grad_norm_([param for param in model.paramaters()], args.grad_clip)
        optimizer.step()


        # Print log info
        if i % args.log_step == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t''{3}\t'.format(epoch, i, len(train_loader), loss/args.batch_size))

        # validate at every val_step
        if i % args.val_step == 0 and i != 0:
            validate(args, val_loader, model)

def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs = encode_data(model, val_loader, opt.log_step, logging.info)

    # caption retrieval

    if opt.data_name == 'wiki':
        (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, measure=opt.measure, npts=1)
        logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                     (r1, r5, r10, medr, meanr))
        # image retrieval
        (r1i, r5i, r10i, medri, meanri) = t2i(img_embs, cap_embs, measure=opt.measure, npts=1)
        logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                     (r1i, r5i, r10i, medri, meanri))

    else:
        (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, measure=opt.measure)
        logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                     (r1, r5, r10, medr, meanr))
        # image retrieval
        (r1i, r5i, r10i, medri, meanri) = t2i(
            img_embs, cap_embs, measure=opt.measure)
        logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                     (r1i, r5i, r10i, medri, meanri))

    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
