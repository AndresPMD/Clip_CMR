import torch

import evaluation_models
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='f30k', help="{coco, f30k}")
    parser.add_argument('--cnn', default='ViT-B/32')
    parser.add_argument('--split', default='test', help='Split, test(1K), testall(5k) (Coco only)')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--workers', default=8, type=int)

    print ("Simple Evaluator of CLIP on Image-Text Matching Datasets \n")
    args = parser.parse_args()
    print ("Arguments: ", args)
    evaluation_models.evalrank(args)
    # elif args.data_name == 'coco':
    #     evaluation_models.evalrank(args.cnn, "pretrain_model/coco/model_coco_1.pth.tar",, data_path='data/', split="testall", fold5=True)
    # else: raise("Dataset Not supported")


if __name__ == "__main__":
    main()
