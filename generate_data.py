import os
import json
import pdb
from tqdm import tqdm

# GENERATES IMG LIST AND CAPTIONS OF F30K AND COCO FOR TESTING

# flickr30k

if os.path.exists('./data/'):
    for file in os.listdir("./data/"):
        os.remove('./data/' + file)

with open ("/SSD/Datasets/Flickr30K/dataset.json" , "r") as fp:
    anns = json.load(fp)

test = [i for i in anns['images'] if i['split'] == 'test']
print("F30k len:", len(test))
for item in test:
    # EVALER requires 5 times image input
        for i in range (5):
            with open ('./data/f30k_images.txt', 'a') as fp:
                fp.write(item['filename'] + '\n')
        
        for caption_dict in item['sentences']:
            with open ('./data/f30k_captions.txt', 'a') as fp:
                fp.write(caption_dict['raw'] + '\n')

# COCO
# LAZY CODING
with open ("/SSD/COCO_raw/caption_datasets/dataset_coco.json" , "r") as fp:
    anns = json.load(fp)

test = [i for i in anns['images'] if i['split'] == 'test']
print("COCO Len: ", len(test))
for item in test:
    for i in range(5):
        with open ('./data/coco_images.txt', 'a') as fp:
            fp.write(item['filename'] + '\n')
        
    for caption_dict in item['sentences']:
        with open ('./data/coco_captions.txt', 'a') as fp:
            fp.write(caption_dict['raw'] + '\n')


print("Complete!")