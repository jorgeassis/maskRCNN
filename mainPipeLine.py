## ------------------------------------------------------------------------
## ------------------------------------------------------------------------
## CHAGE LOG
##
## Implement the Use of GPU
## https://github.com/matterport/Mask_RCNN/issues/1360
##
## ------------------------------------------------------------------------
## ------------------------------------------------------------------------
##
##

# ~/opt/anaconda3/envs/mrcnn/bin/python


## Erase memory objects

for element in dir():
    if element[0:2] != "__":
        del globals()[element]

del element

## Packages and main configuration

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

os.getcwd()

## Root directory of the project
rootDirectory = os.path.abspath("/Volumes/Jellyfish/GDrive/Manuscripts/Convolutional Neural Networks for kelp canopy identification/Git/maskRCNN")

## Import Mask RCNN
sys.path.append(rootDirectory)  # To find local version of the library

## Directory to save logs and trained model
modelDirectory = os.path.join("../../", "logs")

## Dataset Directory

# datasetDirectory = os.path.join(rootDirectory, "Data/")
datasetDirectory = "../../Data/Dataset 3/" # datasetDirectory = "Data/kelpPatches/"

## Local path to trained weights file
weightsFilePath = os.path.join("../../", "mask_rcnn_coco.h5")

# Read the config file
exec(open('customConfig.py').read())
config = mainConfig()
config.display()

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

# Validation dataset
dataset_val = CustomDataset()
dataset_val.load_custom(args.dataset, "train") # val !!!!!!!!!
dataset_val.prepare()

## ------------------------------------------------------------------------
## ------------------------------------------------------------------------

## Visualize dataset

for image_id in dataset_train.image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
    
## ------------------------------------------------------------------------
## ------------------------------------------------------------------------

## Train model

## Download COCO trained weights from Releases if needed
utils.download_trained_weights(weightsFilePath)

# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.

model = modellib.MaskRCNN(mode="training", config=config, model_dir=modelDirectory)

model.load_weights(weightsFilePath, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

from imgaug import augmenters as iaa
augmentationSeq = [iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.Affine(rotate=(0, 360)), iaa.Multiply((0.5, 1.5)) ] #iaa.Affine(scale=(0.5, 1.5)),
                                  
t0 = time.time()

config.LEARNING_RATE = 0.01
config.N_EPOCHS = 10
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=config.N_EPOCHS,
            layers='heads' )

t1 = time.time()
t1-t0

history = model.keras_model.history.history

config.LEARNING_RATE = 0.0001
config.N_EPOCHS = 40
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=config.N_EPOCHS,
            layers='all' ) # , augmentation = iaa.Sequential(augmentationSeq))


t1 = time.time()
t1-t0

new_history = model.keras_model.history.history

## Save weights

# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model.keras_model.save_weights(weightsFilePath)

## ------------------------------------------------------------------------
## ------------------------------------------------------------------------

for k in new_history: history[k] = history[k] + new_history[k]
#if only heads were trained:
#for k in history: history[k] = history[k] 

epochs = range(10)

plt.figure(figsize=(18, 6))

plt.subplot(131)
plt.plot(epochs, history['loss'], label="train loss")
plt.plot(epochs, history['val_loss'], label="valid loss")
plt.legend()
plt.subplot(132)
plt.plot(epochs, history['mrcnn_class_loss'], label="train class loss")
plt.plot(epochs, history['val_mrcnn_class_loss'], label="valid class loss")
plt.legend()
plt.subplot(133)
plt.plot(epochs, history['mrcnn_mask_loss'], label="train mask loss")
plt.plot(epochs, history['val_mrcnn_mask_loss'], label="valid mask loss")
plt.legend()

plt.show()

best_epoch = np.argmin(history["val_loss"]) + 1
print("Best epoch: ", best_epoch)
print("Valid loss: ", history["val_loss"][best_epoch-1])

## ------------------------------------------------------------------------
## ------------------------------------------------------------------------

## Detection

class InferenceConfig(mainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inferenceConfig = InferenceConfig()

# Recreate the model in inference mode

model = modellib.MaskRCNN(mode="inference",
                          config=inferenceConfig,
                          model_dir=modelDirectory)

# Get path of last saved weights

files = glob.glob("../../logs/" + '/**/*.h5', recursive=True)
filesCreatingTime = []

for i in range(len(files)):
    filesCreatingTime.append(os.stat(glob.glob("../../logs/" + '/**/*.h5', recursive=True)[i])[1])

# Get the last
weightsFilePathFinal = files[files.index(max(files))]

# Chose from best_epoch

weightsFilePathFinal = files # Needs to be corrected

# Load trained weights

model.load_weights(weightsFilePathFinal, by_name=True)

## --------------------------
## --------------------------
# Test on a val image

image_id = random.choice(dataset_val.image_ids)
image_id = 15

original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inferenceConfig, image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, figsize=(8, 8))

# Detect

results = model.detect([original_image], verbose=1)

r = results[0]
print(r.keys())
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], figsize=(8, 8))

# Regions identified xmin, ymin, xmax and ymax
r['rois']

# Number of identifyed features
len(r['class_ids'])

# Scores of each feature
r['scores']

# Determine total area (fraction of image)

imageArea = 1 # Change to actual image
totalArea = []
for i in range(r['masks'].shape[-1]):
    mask = r['masks'][:, :, i]
    positive_pixel_count = mask.sum() # assumes binary mask (True == 1)
    h, w = mask.shape # assumes NHWC data format, adapt as needed
    area = positive_pixel_count / (w*h)
    totalArea.append(area)

sum(totalArea)
sum(totalArea) * imageArea

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

totalGeometryaccuracyUnion = totalGeometryObserved.intersection(totalGeometryPredicted)
totalGeometryaccuracyDiff = totalGeometryObserved.difference(totalGeometryPredicted)

totalGeometryObserved.area
totalGeometryPredicted.area
totalGeometryaccuracyUnion.area
totalGeometryaccuracyDiff.area

# Accuracy indexes
# https://towardsdatascience.com/how-accurate-is-image-segmentation-dd448f896388

# Jaccard’s Index (Intersection over Union, IoU)

index = totalGeometryaccuracyUnion.area / ( totalGeometryObserved.area + totalGeometryPredicted.area - totalGeometryaccuracyUnion.area )
index

# Dice Coefficient

index = 2 * totalGeometryaccuracyUnion.area / ( 2 *  totalGeometryaccuracyUnion.area + ( totalGeometryObserved.area + totalGeometryPredicted.area - totalGeometryaccuracyUnion.area ))
index

# use descartes to create the matplotlib patches
ax = plt.gca()
ax.add_patch(descartes.PolygonPatch(totalGeometryaccuracyUnion, fc='GREEN', ec='GREEN', alpha=0.5))
ax.add_patch(descartes.PolygonPatch(totalGeometryaccuracyDiff, fc='RED', ec='RED', alpha=0.5))

# control display
ax.set_xlim(0, 800); ax.set_ylim(0, 800)
ax.set_aspect('equal')
plt.show()

## --------------------------
## --------------------------
# Test on a external image

externalImagePath = "../../Data/Dataset 3/train/LC08_040037_20140914_27.jpg"

# display image

plt.clf()
img = mpimg.imread(externalImagePath)
imgplot = plt.imshow(img)
plt.show()

img = load_img(externalImagePath)
img = img_to_array(img)
results = model.detect([img], verbose=1)
r = results[0]
visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names, r['scores'], figsize=(8, 8))

r['scores']

## ------------------------------------------------------------------------
## ------------------------------------------------------------------------

## Tunning parameters with cross-validation
## https://towardsdatascience.com/3-ways-to-tune-hyperparameters-of-machine-learning-models-with-python-cda64b62e0ac

image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    index =
    APs.append(index)

print("mAP: ", np.mean(APs))
