import torch

import evaluation_models
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='f30k', help="{coco, f30k}")
    parser.add_argument('--cnn', default='ViT-B/32')

    print ("Simple Evaluator of CLIP on Image-Text Matching Datasets \n")
    if args.data_name == 'f30k':
        print('Evaluation on Flickr30K:')
        evaluation_models.evalrank("runs/flickr_VSRN/model_best.pth.tar", "runs/flickr_VSRN/model_best.pth.tar",
                           data_path='/SSD/Datasets/Precomp_features/data', split="test", fold5=False)
    elif args.data_name == 'coco':
        evaluation_models.evalrank("pretrain_model/coco/model_coco_1.pth.tar", "pretrain_model/coco/model_coco_2.pth.tar", data_path='data/', split="testall", fold5=True)
    else: raise("Dataset Not supported")


if __name__ == "__main__":
    main()
