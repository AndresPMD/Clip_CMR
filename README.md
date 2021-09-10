# CLIP based Image Text Matching 

This projects employs CLIP (Paper: https://arxiv.org/pdf/2103.00020.pdf) as a backbone to perform image-text retrieval.

The results obtained from the publicly available weights from CLIP do not yield good results on Flickr30K and MSCOCO.
The public model achieves:

<table>
  <tr>
    <td></td>
    <td colspan="3">Image-to-Text</td>
    <td colspan="3">Text-to-Image</td>
    <td></td>
  </tr>
  <tr>
    <td>Dataset</td>
    <td>R@1</td>
    <td>R@5</td>
    <td>R@10</td>
    <td>R@1</td>
    <td>R@5</td>
    <td>R@10</td>

  </tr>
  <tr>
    <td>MSCOCO-1K</td>
    <td>26.1</td>
    <td>64.6</td>
    <td>81.2</td>
    <td>48.0</td>
    <td>77.5</td>
    <td>88.2</td>
  </tr>
  <tr>  
    <td>Flickr30k</td>
    <td>36.0</td>
    <td>71.9</td>
    <td>83.4</td>
    <td>55.8</td>
    <td>80.7</td>
    <td>88.3</td>
  </tr>
</table>

This project trains a non-linearity on top of CLIP features as a finetuning step to improve the learned representations.
The added non-linear probe performs significantly better when fine-tuned in these datasets.


## Install

Please follow the installation requirements from the oficial CLIP repository:

https://github.com/openai/CLIP

## Generate Data

This model requires to generate two txt files that include the images and the captions to be used by the model.

Run:

    $ python generate_data.py


# Train

Modify the data_path accordingly in the dataloader. 

To train in Flickr30K run:

    $ python train.py --data_name f30k --logger_name runs/clip_ft_f30k

To train in MSCOCO run:

    $ python train.py --data_name coco --logger_name runs/clip_ft_coco

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)