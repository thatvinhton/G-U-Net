## UNet
Most code is learned from: https://github.com/DrSleep/tensorflow-deeplab-resnet

## Requirements

- Use anaconda3 to replicate the python environment. Details can be found here: https://conda.io/docs/user-guide/install/download.html
- After install anaconda3, clone this directory to computer by running this command in terminal: `git clone https://tvton@bitbucket.org/tvton/unet.git`
- Change to cloned directory.
- To create the replicated python environment: `conda env create -f environment.yml`
- To activate replicated environment: `source activate bio`

This code is tested on gpu: TITAN X 12GB. Other documents can be found at: https://conda.io/docs/user-guide/tasks/manage-environments.html 


## Inference
- Download the trained model and extract in the root of cloned directory: https://drive.google.com/open?id=1cpwDYvhOvdaTnssPHoRcEWgQ8cjSStqV

- To run the network on single image, follow this instruction:
```
python inference.py --img-link=<link to image> --checkpoints=<link to directory containing trained model>
```
For example:
```
python inference.py --img-link=./TCGA-18-5592-01Z-00-DX1.tif --checkpoints=./model
```
The result is the new image with name 'output.png'.

- To run the network on whole directory containing a set of image, follow this instruction:
```
python inference.py --img-link=<link to image directory> --checkpoints=./model
```

## Short description

1. scripts/dataloader.py: Some code for data loader.
2. scripts/model.py: Contain classes which define the structure of network.
3. scripts/network.py: Implement some basic layers or blocks for network.
4. scripts/tensorboard_logging.py: For logging the training process.
5. train.py: Training code
6. test.py: Test code (using 256x256 patch as input to check the output of the trained model).
7. inference.py: Inference code using test time augmentation - TTA. Input will be an image/directory containing images of size 1000x1000.
7. inference_no_tta.py: Inference code without using test time augmentation - TTA. Input will be an image/directory containing images of size 1000x1000.

