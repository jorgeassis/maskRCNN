## ------------------------------------------------------------------------
## ------------------------------------------------------------------------
##
##

# ~/opt/anaconda3/envs/mrcnn/bin/python
# /home/jorgeassis/miniconda3/envs/mrcnn/bin/python

## Erase memory objects

for element in dir():
    if element[0:2] != "__":
        del globals()[element]

del element

## Packages and main configuration

from customDatasetClass import CustomDataset2
import os
import glob

import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imantics import Polygons, Mask
import argparse
import json
import skimage.draw
import skimage.viewer

from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from shapely import geometry
import geopandas as gpd

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from pandas import DataFrame
from pandas import concat
import pandas as pd

os.chdir("/media/Jellyfish/Backup [Ongoing studies]/Mask RCNN for Kelp Detection/Code/maskRCNN/")
os.getcwd()

## Root directory of the project
rootDirectory = os.path.abspath("/media/Jellyfish/Backup [Ongoing studies]/Mask RCNN for Kelp Detection/Code/maskRCNN/")

## Import Mask RCNN
sys.path.append(rootDirectory)  # To find local version of the library

## Directory to save logs and trained model
modelDirectory = os.path.join("../../", "Experiments/J08")

## Dataset Directory

# datasetDirectory = os.path.join(rootDirectory, "Data/")
datasetDirectory = "/media/Jellyfish/Backup [Ongoing studies]/Mask RCNN for Kelp Detection/Annotations Data/Final Data/" # datasetDirectory = "Data/kelpPatches/"

## Local path to trained weights file
weightsFilePath = os.path.join("../../", "mask_rcnn_coco.h5")

# Read the config file
exec(open('customConfig.py').read())
config = mainConfig()
config.display()

class InferenceConfig(mainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inferenceConfig = InferenceConfig()

## ------------------------------------------------------------------------
## ------------------------------------------------------------------------

# Read custom dataset class
exec(open('customDatasetClass.py').read())

parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect custom class.')

parser.add_argument('--dataset', required=False,
                    metavar=datasetDirectory,
                    default=datasetDirectory,
                    help='Directory of the custom dataset')

args = parser.parse_args()

# Training dataset.
dataset_train = CustomDataset()
dataset_train.load_custom(args.dataset, "train")
dataset_train.prepare()

# Test dataset
dataset_val = CustomDataset()
dataset_val.load_custom(args.dataset, "test")
dataset_val.prepare()

# Load validation dataset
dataset_independent = CustomDataset2()
dataset_independent.load_custom(args.dataset, "val")
dataset_independent.prepare()

## ---------------------
## ---------------------

imageArea = 1 # Change to actual image

resultsArea = []

for i in range(len(dataset_train._image_ids)):   

    image_id = i

    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_train, inferenceConfig, image_id, use_mini_mask=False)

    totalGeometry = []
    for i in range(gt_mask.shape[-1]):
        polygons = Mask(gt_mask[:, :, i]).polygons()
        polygons = polygons.points[:][0]
        if len(polygons) <= 2:
            continue
        polygonF = Polygon(polygons)
        totalGeometry.append(polygonF)

    totalGeometryPredicted = cascaded_union(totalGeometry)

    if str(type(totalGeometryPredicted)) == "<class 'shapely.geometry.polygon.Polygon'>":
        nPatches = 1

    if str(type(totalGeometryPredicted)) != "<class 'shapely.geometry.polygon.Polygon'>":
        nPatches = len(totalGeometryPredicted)

    resultsArea.append(
        {
            'Detections': nPatches,
            'Area': totalGeometryPredicted.area * imageArea,
        }
    )

resultsArea = DataFrame(resultsArea)
resultsArea

resultsArea.to_csv('../../dataset_train_traits.csv', index = False, header=True)

