# High Resolution Canopy Height Maps

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

Jamie Tolan,
Hung-I Yang, 
Benjamin Nosarzewski,
Guillaume Couairon, 
Huy V. Vo, 
John Brandt, 
Justine Spore, 
Sayantan Majumdar, 
Daniel Haziza, 
Janaki Vamaraju, 
Théo Moutakanni, 
Piotr Bojanowski, 
Tracy Johns, 
Brian White, 
Tobias Tiecke, 
Camille Couprie

[[`Paper`](https://doi.org/10.1016/j.rse.2023.113888)][[`ArxiV [same content]`](https://arxiv.org/abs/2304.07213)] [[`Blog`](https://research.facebook.com/blog/2023/4/every-tree-counts-large-scale-mapping-of-canopy-height-at-the-resolution-of-individual-trees/)] [[`BibTeX`](#citing-HighResCanopyHeight)]



PyTorch implementation and pretrained models for High resolution Canopy Height Prediction inference. For details, see the paper: 
**[Very high resolution canopy height maps from RGB imagery using self-supervised  vision transformer and convolutional decoder trained on Aerial Lidar](https://arxiv.org/abs/2304.07213)**.

In collaboration with the Physical Modeling and Sustainability teams at Meta, and the World Resource Institute, we applied [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) to the canopy height map (CHM) prediction problem. We used this technique to pretrain a backbone on about 18 millions satellite images around the globe. We then trained a CHM predictor on a modest sized training dataset covering a few thousand square kilometers of forests in the United States, with this backbone as feature extractor. 
We demonstrate in our paper quantitatively and qualitatively the advantages of large-scale self-supervised learning, the versatility of obtained representations allowing generalization to different geographic regions and input imagery.

The maps obtained with this model are available at https://wri-datalab.earthengine.app/view/submeter-canopyheight. 
  

![alt text](https://github.com/facebookresearch/HighResCanopyHeight/blob/main/fig_0_1_2.png)

Fork by Jón Tómasson for easier predictions of new image data.

## Setting up environment

Example of successful environment creation for inference using [Anaconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).

```
conda create -n hrch python=3.9 -y
conda activate hrch
conda install numpy==1.26.4 pytorch-lightning==1.7 pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 torchmetrics==0.11.4 pillow==11.0.0 pandas==2.2.3 matplotlib==3.9.4 -c pytorch -c nvidia -c conda-forge

```



### Data
To get the pretrained model you need the [AWS-CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). 
To prepare the data, in the cloned repository, run these commands:

```
aws s3 --no-sign-request cp --recursive s3://dataforgood-fb-data/forests/v1/models/ .
unzip data.zip
rm data.zip
```
The data.zip also contains an image folder with some aerial images.


## Running the inference

To run the inference you need a .csv file with a list of images and an identifier for each image. An example is provided in ./data/my_data.csv.
The images are cropped such that the used area is a 256x256 square area starting from the top left corner of the image.

The following to commands are equvalent. These are the default arguments. 

```
 python inference.py --csv ./data/my_data.csv --image_dir ./data/images/ --name output_inference

 python inference.py
```

The output will be in the output_inference folder. For each input image there will be a generated visualisation image of the height map and 
the height map stored as a numpy array you can load the height maps back into Python with:
```
import numpy as np

a = np.load('filename.npy')
```



## License

HighResCanopyHeight code and model weights are released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details.

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citing HighResCanopyHeight

If you find this repository useful, please consider giving a star :star: and citation :t-rex::

```
@article{tolan2024very,
  title={Very high resolution canopy height maps from RGB imagery using self-supervised vision transformer and convolutional decoder trained on aerial lidar},
  author={Tolan, Jamie and Yang, Hung-I and Nosarzewski, Benjamin and Couairon, Guillaume and Vo, Huy V and Brandt, John and Spore, Justine and Majumdar, Sayantan and Haziza, Daniel and Vamaraju, Janaki and others},
  journal={Remote Sensing of Environment},
  volume={300},
  pages={113888},
  year={2024}
}
```

