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

os.getcwd()

## Root directory of the project
rootDirectory = os.path.abspath("/media/Bathyscaphe/Mask RCNN for Kelp Detection/Code/maskRCNN/")

## Import Mask RCNN
sys.path.append(rootDirectory)  # To find local version of the library

## Directory of trained model
modelDirectory = os.path.join("../../", "Experiments/J02")

# Read the config file
exec(open('customConfig.py').read())
config = mainConfig()
config.display()

## --------------------------------------------------------------------------
## --------------------------------------------------------------------------
# Reproduce the model

parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect custom class.')

parser.add_argument('--dataset', required=False,
                    metavar=datasetDirectory,
                    default=datasetDirectory,
                    help='Directory of the custom dataset')

args = parser.parse_args()

class InferenceConfig(mainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inferenceConfig = InferenceConfig()

# Recreate the model in inference mode

inferenceConfig.display()
inferenceConfig.DETECTION_MIN_CONFIDENCE = 0.5
inferenceConfig.DETECTION_NMS_THRESHOLD = 0.5 # 0.9 in model fitting
inferenceConfig.DETECTION_MAX_INSTANCES = 1000

model = modellib.MaskRCNN(mode="inference",
                          config=inferenceConfig,
                          model_dir=modelDirectory)

# Get path of last saved weights

files = glob.glob(modelDirectory + '/**/*.h5', recursive=True)
filesCreatingTime = []

for i in range(len(files)):
    filesCreatingTime.append(os.stat(glob.glob(modelDirectory + '/**/*.h5', recursive=True)[i])[1])

# Chose from best_epoch ****

best_epoch = 30 - 1
weightsFilePathFinal = glob.glob(modelDirectory + '/**/*' + str(best_epoch) + '.h5', recursive=True)[0]

model.load_weights(weightsFilePathFinal, by_name=True)

## --------------------------------------------------------------------------
## --------------------------------------------------------------------------
# Detect on validation dataset images

exec(open('customDatasetClass.py').read())

parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect custom class.')

parser.add_argument('--dataset', required=False,
                    metavar=datasetDirectory,
                    default=datasetDirectory,
                    help='Directory of the custom dataset')

args = parser.parse_args()

datasetDirectory = "/media/Bathyscaphe/Mask RCNN for Kelp Detection/Annotations Data/Final Data/" # datasetDirectory = "Data/kelpPatches/"

# Load validation dataset
dataset_independent = CustomDataset2()
dataset_independent.load_custom(args.dataset, "val")
dataset_independent.prepare()

# Detect for an image of the val dataset

image_id = 1 # 1 6 7 9

original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_independent, inferenceConfig, image_id, use_mini_mask=False)

# dataset_independent.class_names
visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,['BG', ''], figsize=(12,12), title="Mask RCNN Kelp Annotation", captions=None, scores=None)

results = model.detect([original_image], verbose=1)
r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],['BG', ''], figsize=(12,12), title="Mask RCNN Kelp Detection", captions=None, scores=None)

## --------------------------------------------------------------------------
## --------------------------------------------------------------------------
# Detect on external image

imagePath = "/media/Bathyscaphe/Mask RCNN for Kelp Detection/Annotations Data/Final Data/val/LC08_037041_20180515_11.jpg"
original_image = load_img(imagePath)
original_image = img_to_array(original_image)

results = model.detect([original_image], verbose=1)

r = results[0]
print(r.keys())
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], ['BG', ''], r['scores'], figsize=(8, 8))

# Regions identified xmin, ymin, xmax and ymax
r['rois']

# Number of identifyed features
len(r['class_ids'])

# Scores of each feature
r['scores']

# ----------------------------------------------------
# ----------------------------------------------------

# Determine total area (fraction of image)

totalGeometry = []
for i in range(r['masks'].shape[-1]):
    polygons = Mask(r['masks'][:, :, i]).polygons()
    polygons = polygons.points[:][0]
    polygonF = Polygon(polygons)
    totalGeometry.append(polygonF)

totalGeometryPredicted = cascaded_union(totalGeometry)

boundaryP = gpd.GeoSeries(totalGeometryPredicted)
boundaryP.plot(color = 'red')
plt.show()
plt.close()

# To shapefile
# use the geature loop in case you polygon is a multipolygon
features = [i for i in range(len(totalGeometryPredicted))]
# add crs using wkt or EPSG to have a .prj file
gdr = gpd.GeoDataFrame({'feature': features, 'geometry': totalGeometryPredicted}) #, crs='EPSG:4326)
gdr.to_file("Poly.shp")

# Actual observed mask

totalGeometry = []
for i in range(gt_mask.shape[-1]):
    polygons = Mask(gt_mask[:, :, i]).polygons()
    polygons = polygons.points[:][0]
    polygonF = Polygon(polygons)
    totalGeometry.append(polygonF)

totalGeometryObserved = cascaded_union(totalGeometry)

boundaryO = gpd.GeoSeries(totalGeometryObserved)
boundaryO.plot(color = 'green')
plt.show()
plt.close()

# ----------------------------------------------------

totalGeometryPredicted = totalGeometryPredicted.buffer(0)
totalGeometryObserved = totalGeometryObserved.buffer(0)

totalGeometryaccuracyIntersect = totalGeometryObserved.intersection(totalGeometryPredicted)
totalGeometryaccuracyDiff = totalGeometryObserved.difference(totalGeometryPredicted)

totalGeometryObserved.area
totalGeometryPredicted.area
totalGeometryaccuracyIntersect.area
totalGeometryaccuracyDiff.area

# Accuracy indexes
# https://towardsdatascience.com/how-accurate-is-image-segmentation-dd448f896388

# Jaccard’s Index (Intersection over Union, IoU)

index = totalGeometryaccuracyIntersect.area / ( totalGeometryaccuracyIntersect.area  + totalGeometryaccuracyDiff.area )
index

# Dice Coefficient

index = 2 * totalGeometryaccuracyIntersect.area / ( totalGeometryaccuracyIntersect.area  + totalGeometryaccuracyIntersect.area  + totalGeometryaccuracyDiff.area )
index

# use descartes to create the matplotlib patches
import descartes
ax = plt.gca()
ax.add_patch(descartes.PolygonPatch(totalGeometryaccuracyDiff, fc='BLACK', ec='BLACK', alpha=0.5))
ax.add_patch(descartes.PolygonPatch(totalGeometryaccuracyIntersect, fc='GREEN', ec='GREEN', alpha=0.5))

# control display
ax.set_xlim(0, 800); ax.set_ylim(0, 800)
ax.set_aspect('equal')
plt.show()

## -----------------------------------------------------------------------
## -----------------------------------------------------------------------
## Detect for a loop

externalImagePath = "/media/Bathyscaphe/Mask RCNN for Kelp Detection/TimeSeriesData/Landsat_037041_Tiles/"
externalImageFiles = glob.glob(externalImagePath + '**.jpg')
tiles = ['_3.','_4.', '_11.', '_19.']

externalImageFiles = [i for i in externalImageFiles if any(b in i for b in tiles)]
externalImageFiles

resultsArea = []

for i in range(len(externalImageFiles)):   

    imageName = externalImageFiles[i].replace(externalImagePath , "")
    imageName = imageName.replace(".JPG", "")
    imageName = imageName.replace(".jpg", "")

    imagePath = externalImageFiles[i]

    img = load_img(imagePath)
    img = img_to_array(img)
    results = model.detect([img], verbose=1)
    r = results[0]
    # visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],dataset_independent.class_names, r['scores'], figsize=(8, 8))

    imageArea = 1 # Change to actual image

    totalGeometry = []
    for i in range(r['masks'].shape[-1]):
        polygons = Mask(r['masks'][:, :, i]).polygons()
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
            'Image': imageName,
            'Detections': nPatches,
            'Area': totalGeometryPredicted.area * imageArea,
        }
    )

resultsArea = DataFrame(resultsArea)
resultsArea
resultsArea.to_csv(externalImagePath + '/areaEstimates.csv', index = False, header=True)
