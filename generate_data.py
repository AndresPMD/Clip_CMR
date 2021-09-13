import os
import json
import pdb
from tqdm import tqdm
import re

# GENERATES IMG LIST AND CAPTIONS OF F30K AND COCO FOR TESTING


def generate_files(items, split_name, dataset):
    images = []
    captions = []
    for item in items:
        # EVALER requires 5 times image input
        for i in range (5):
            images.append(item['filename'])

        # Captions
        for idx, caption_dict in enumerate(item['sentences']):
            if idx >= 5:
                break
            else:
                cleaned_string = re.sub(r'[^A-Za-z0-9 ]+', '', caption_dict['raw'])
                captions.append(cleaned_string)

    # Write Files
    with open('./data/{}_{}_ims.txt'.format(dataset, split_name), 'w') as fp:
        for i in images:
            fp.write(i + '\n')

    with open('./data/{}_{}_caps.txt'.format(dataset, split_name), 'w') as fp:
        for c in captions:
            fp.write(c.strip() + '\n')


def main ():
    if os.path.exists('./data/'):
        for file in os.listdir("./data/"):
            os.remove('./data/' + file)
    with open ("/SSD/Datasets/Flickr30K/dataset_flickr30k.json", "r") as fp:
        anns = json.load(fp)

    train = [i for i in anns['images'] if i['split'] == 'train']
    print("F30k Train len:", len(train))
    generate_files(train, 'train', 'f30k')

    val = [i for i in anns['images'] if i['split'] == 'val']
    print("F30k Val len:", len(val))
    generate_files(val, 'val', 'f30k')

    test = [i for i in anns['images'] if i['split'] == 'test']
    print("F30k Test len:", len(test))
    generate_files(test, 'test', 'f30k')

    # COCO
    # LAZY CODING
    with open ("/SSD/COCO_raw/caption_datasets/dataset_coco.json", "r") as fp:
        anns = json.load(fp)

    train = [i for i in anns['images'] if i['split'] == 'train' or i['split'] == 'restval']
    print("COCO train Len: ", len(train))
    generate_files(train, 'train', 'coco')

    val = [i for i in anns['images'] if i['split'] == 'val']
    print("COCO val Len: ", len(val))
    generate_files(val, 'val', 'coco')

    test = [i for i in anns['images'] if i['split'] == 'test']
    print("COCO test Len: ", len(test))
    generate_files(test, 'test', 'coco')

    print("Complete!")

if __name__ == '__main__':
    main()