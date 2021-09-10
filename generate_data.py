import os
import json
import pdb
from tqdm import tqdm

# GENERATES IMG LIST AND CAPTIONS OF F30K AND COCO FOR TESTING

# Flickr30k
def generate_files(items, split_name, dataset):
    for item in items:
        # EVALER requires 5 times image input
        for i in range (5):
            with open ('./data/{}_{}_ims.txt'.format(dataset, split_name), 'a') as fp :
                fp.write(item['filename'] + '\n')
        
        for idx, caption_dict in enumerate(item['sentences']):
            if idx >= 5: break
            with open ('./data/{}_{}_caps.txt'.format(dataset, split_name), 'a') as fp:
                fp.write(caption_dict['raw'].strip() + '\n')

def main ():
    if os.path.exists('./data/'):
        for file in os.listdir("./data/"):
            os.remove('./data/' + file)
    with open ("/SSD/Datasets/Flickr30K/dataset_flickr30k.json" , "r") as fp:
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
    with open ("/SSD/COCO_raw/caption_datasets/dataset_coco.json" , "r") as fp:
        anns = json.load(fp)

    train = [i for i in anns['images'] if i['split'] == 'train']
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