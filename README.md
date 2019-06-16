## Enhanced Rotation-Equivariant U-Net for Nuclear Segmentation
Chidester, Benjamin and Ton, That-Vinh and Tran, Minh-Triet and Ma, Jian and Do, Minh N.  

Computer Vision for Microscopy Image Analysis, CVPRW 2019. <a href="http://openaccess.thecvf.com/content_CVPRW_2019/papers/CVMI/Chidester_Enhanced_Rotation-Equivariant_U-Net_for_Nuclear_Segmentation_CVPRW_2019_paper.pdf">[PDF]</a>

## Requirements

- Use anaconda3 to replicate the python environment.
- After install anaconda3, clone this directory: `https://github.com/thatvinhton/G-U-Net.git`
- Change to cloned directory.
- Create the replicated python environment: `conda env create -f environment.yml`
- Activate replicated environment: `source activate bio`

This code is tested on gpu: TITAN V 12GB. Other documents can be found at: https://conda.io/docs/user-guide/tasks/manage-environments.html 

## Pre-processing
Images should be pre-processed by method proposed in "*A.Vahadane, T.Peng, S.Albarqouni, M.Baust, K.Steiger, A.M.Schlitter, A.Sethi, I.Esposito, and N.Navab.* **Structure-preserved color normalization forhistological images.** In ISBI, pages 1012–1015, April 2015."

We use image **TCGA-18-5592-01Z-00-DX1.tif** as target image and convert all other images to its color space.
Lambda = 0.1 is used as recommendation. 


## Inference
- Download and extract the trained model: https://drive.google.com/file/d/16km15kPOgLyIWZhkq_W3qMdrVvsVgAHf/view?usp=sharing

- The current inference code works on image of size 1000x1000.

- To run the G-U-Net on single image, follow this instruction:
```
python g_inference.py --img-link=<link to image> --checkpoints=<link to directory containing trained model>
```
The result is the new image with name 'output.png'.

- To run the G-U-Net on whole directory containing a set of image, follow this instruction:
```
python g_inference.py --img-link=<link to image directory> --checkpoints=<link to directory containing trained model> --result-dir=<link to directory containing results>
```

## Post-processing

The results created from inference step should be post-processed to create final index.
Change the result directory (from previous step) as input in *postProcessing.py* and run to create final results. 

## Short description

1. scripts/dataloader.py: Define Tensorflow's dataloader.
2. scripts/model.py: Define U-Net architecture.
3. scripts/network.py: Implement some basic layers or blocks used in network.
4. scripts/tensorboard_logging.py: For logging the training process.
5. train.py: Code used to train ordinal U-Net (see required arguments for further information).
6. g_train.py: Code used to train G-U-Net (see required arguments for further information).
7. inference.py: Create prediction from U-Net.
8. g_inference.py: Create prediction from G-U-Net.

## Acknowledgements

1. Most code is learned from: https://github.com/DrSleep/tensorflow-deeplab-resnet
2. The g-conv layers are tensorflow implementation from this project: https://github.com/tscohen/GrouPy
3. The g-batchnorm layers are implemented based on this project: https://github.com/ZJULearning/DREN

## Citation
Please cite our work if it helps your research/project:
```
@InProceedings{Chidester_2019_CVPR_Workshops,
author = {Chidester, Benjamin and Ton, That-Vinh and Tran, Minh-Triet and Ma, Jian and Do, Minh N.},
title = {Enhanced Rotation-Equivariant U-Net for Nuclear Segmentation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}
```
