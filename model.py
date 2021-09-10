
import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())
# Add Linear probing to CLIP as backbone

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

class Clip_Linear(nn.Module):
    def __init__(self, model_clip, args):
        super(Clip_Linear, self).__init__()
        self.model_clip = model_clip
        self.embed_size = args.embed_size

        self.grad_clip = args.grad_clip

        # Add two FC for img and txt
        self.img_bn1 = nn.BatchNorm1d(512)
        self.img_fc1 = nn.Linear(512, self.embed_size)
        self.img_bn2 = nn.BatchNorm1d(self.embed_size)
        self.img_fc2 = nn.Linear(self.embed_size, self.embed_size)
        
        self.txt_bn1 = nn.BatchNorm1d(512)
        self.txt_fc1 = nn.Linear(512, self.embed_size)
        self.txt_bn2 = nn.BatchNorm1d(self.embed_size)
        self.txt_fc2 = nn.Linear(self.embed_size, self.embed_size)


    def forward(self, images, captions, *args):
        """One training step given images and captions.
        """
        images = self.model_clip.encode_image(images)
        images = images.float()
        img_emb = F.gelu(self.img_fc1(self.img_bn1(images)))
        img_emb = self.img_fc2(self.img_bn2(img_emb))
        img_emb = l2norm(img_emb)

        captions = self.model_clip.encode_text(captions)
        captions = captions.float()
        cap_emb = F.gelu(self.txt_fc1(self.txt_bn1(captions)))
        cap_emb = self.txt_fc2(self.txt_bn2(cap_emb))
        cap_emb = l2norm(cap_emb)

        return img_emb, cap_emb
