import torch

import evaluation_models
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='f30k', help="{coco, f30k}")
    parser.add_argument('--cnn', default='RN50x16', help='[RN50, RN101, RN50x4, RN50x16, ViT-B/32, ViT-B/16]')
    parser.add_argument('--split', default='test', help='Split, test(1K), testall(5k) (Coco only)')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)

    print ("Simple Evaluator of CLIP on Image-Text Matching Datasets \n")
    args = parser.parse_args()
    print ("Arguments: ", args)
    evaluation_models.evalrank(args)

if __name__ == "__main__":
    main()
