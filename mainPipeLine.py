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

## Directory to save logs and trained model
modelDirectory = os.path.join("../../", "Experiments/J04")

## Dataset Directory

# datasetDirectory = os.path.join(rootDirectory, "Data/")
datasetDirectory = "/media/Bathyscaphe/Mask RCNN for Kelp Detection/Annotations Data/Final Data/" # datasetDirectory = "Data/kelpPatches/"

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

# Test dataset
dataset_val = CustomDataset()
dataset_val.load_custom(args.dataset, "test")
dataset_val.prepare()

## ------------------------------------------------------------------------
## ------------------------------------------------------------------------

## Visualize dataset

for image_id in dataset_train.image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
    

for image_id in dataset_val.image_ids:
    image = dataset_val.load_image(image_id)
    mask, class_ids = dataset_val.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_val.class_names)
    
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

# Train heads

start = time.time()

config.N_EPOCHS = 10

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=config.N_EPOCHS,
            layers='heads' )

# Save loss to file

history = model.keras_model.history.history
historyDF = DataFrame(history)
historyDF.to_csv(modelDirectory + '/historyExperimentHeads.csv', index = False, header=True)

## Model all layers

from imgaug import augmenters as iaa
augmentationSeq = [iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.Affine(rotate=(0, 360)), iaa.Multiply((0.5, 1.5)) ] #iaa.Affine(scale=(0.5, 1.5)),
     
config.N_EPOCHS = 50 # 100 final version

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=config.N_EPOCHS,
            layers='all', augmentation = iaa.Sequential(augmentationSeq))
            
end = time.time()
print(f"Runtime of the program is {(end - start) / 60} minutes")

## ------------------

new_history = model.keras_model.history.history
for k in new_history: 
    history[k] = history[k] + new_history[k]

historyDF = DataFrame(history)
historyDF.to_csv(modelDirectory + '/historyExperimentAll.csv', index = False, header=True)

# Plot loss

plt.close()
plt.close(f)

epochs = range(model.epoch)
f = plt.figure(figsize=(18, 6))

plt.subplot(131)
plt.plot(epochs, history['loss'], label="train loss")
plt.plot(epochs, history['val_loss'], label="valid loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss')
ax = plt.gca()
ax.get_legend().remove()
ax.set_xticks(range(model.epoch))

plt.subplot(132)
plt.plot(epochs, history['mrcnn_bbox_loss'], label="train class loss") 
plt.plot(epochs, history['val_rpn_bbox_loss'], label="valid class loss")
plt.title('BBox loss')
plt.xlabel('Epoch')
ax = plt.gca()
ax.get_legend().remove()
ax.set_xticks(range(model.epoch))

plt.subplot(133)
plt.plot(epochs, history['mrcnn_mask_loss'], label="train loss")
plt.plot(epochs, history['val_mrcnn_mask_loss'], label="valid loss")
plt.title('Mask loss')
plt.xlabel('Epoch')
ax = plt.gca()
ax.get_legend().remove()
ax.set_xticks(range(model.epoch))
plt.legend(loc = "upper left")
plt.legend()

plt.show()
f.savefig(modelDirectory + '/lossPlot.pdf', bbox_inches='tight')
plt.close()
plt.close(f)

## ------------------------
##
## conda activate mrcnn
## tensorboard --logdir logs/ # Logs is the folder where the experiments are 
## http://10.36.5.144:6006/
##
## ------------------------

# Get best epoch

best_epoch = np.argmin(history["val_loss"])
print("Best epoch: ", best_epoch + 1)
print("Valid loss: ", history["val_loss"][best_epoch])

lossResultsBestEpoch = DataFrame(history)
df2 = DataFrame([best_epoch + 1], index=["Best epoch"])
df3 = lossResultsBestEpoch.loc[best_epoch]
pd.concat([df2, df3]).to_csv(modelDirectory + '/lossesBestEpoch.csv', index = True, header=False)

## ------------------------

## Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model.keras_model.save_weights(weightsFilePath)

## ------------------------------------------------------------------------
## ------------------------------------------------------------------------

## Detection

# Load validation dataset
dataset_independent = CustomDataset2()
dataset_independent.load_custom(args.dataset, "val")
dataset_independent.prepare()

## -----------------

class InferenceConfig(mainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inferenceConfig = InferenceConfig()

# Recreate the model in inference mode

inferenceConfig.display()
inferenceConfig.DETECTION_MIN_CONFIDENCE = 0
inferenceConfig.DETECTION_NMS_THRESHOLD = 0.9

model = modellib.MaskRCNN(mode="inference",
                          config=inferenceConfig,
                          model_dir=modelDirectory)

# Load from external file 

# weightsFilePathFinal = '/media/Bathyscaphe/Mask RCNN for Kelp Detection/logs/kelp20210528T1202/mask_rcnn_kelp_0028.h5'

# Get path of last saved weights

files = glob.glob(modelDirectory + '/**/*.h5', recursive=True)
filesCreatingTime = []

for i in range(len(files)):
    filesCreatingTime.append(os.stat(glob.glob(modelDirectory + '/**/*.h5', recursive=True)[i])[1])

# Get the last

weightsFilePathFinal = files[files.index(max(files))]

# Chose from best_epoch ****
 
weightsFilePathFinal = glob.glob(modelDirectory + '/**/*' + str(best_epoch) + '.h5', recursive=True)[0]

## --------------------------

# Load trained weights

model.load_weights(weightsFilePathFinal, by_name=True)

## ------------------------------------------------------------------------
## ------------------------------------------------------------------------

## Loop to get accuracy values for different threholds of confidence

resultsDFThreshold = []

for threshold in np.arange(0, 1, 0.1): 

    inferenceConfig.DETECTION_MIN_CONFIDENCE = round(threshold, 2)
    
    model = modellib.MaskRCNN(mode="inference",
                            config=inferenceConfig,
                            model_dir=modelDirectory)
                                                
    model.load_weights(weightsFilePathFinal, by_name=True)

    resultsDF = []
    numberValImages = len(glob.glob(datasetDirectory + '/val/*.jpg', recursive=True))

    for image_id in range(numberValImages):

        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_independent, inferenceConfig, image_id, use_mini_mask=False)

        # Detect

        results = model.detect([original_image], verbose=1)
        r = results[0]

        totalGeometry = []
        for i in range(r['masks'].shape[-1]):
            polygons = Mask(r['masks'][:, :, i]).polygons()
            polygons = polygons.points[:][0]
            if len(polygons) <= 2:
                continue
            polygonF = Polygon(polygons)
            totalGeometry.append(polygonF)

        totalGeometryPredicted = cascaded_union(totalGeometry)

        boundaryP = gpd.GeoSeries(totalGeometryPredicted)
        # boundaryP.plot(color = 'red')
        # plt.show()

        totalGeometry = []
        for i in range(gt_mask.shape[-1]):
            polygons = Mask(gt_mask[:, :, i]).polygons()
            polygons = polygons.points[:][0]
            if len(polygons) <= 2:
                continue
            polygonF = Polygon(polygons)
            totalGeometry.append(polygonF)

        totalGeometryObserved = cascaded_union(totalGeometry)

        boundaryO = gpd.GeoSeries(totalGeometryObserved)
        # boundaryO.plot(color = 'green')
        # plt.show()

        totalGeometryPredicted = totalGeometryPredicted.buffer(0)
        totalGeometryObserved = totalGeometryObserved.buffer(0)

        totalGeometryaccuracyIntersect = totalGeometryObserved.intersection(totalGeometryPredicted)
        totalGeometryaccuracyDiff = totalGeometryObserved.difference(totalGeometryPredicted)

        indexJaccard = totalGeometryaccuracyIntersect.area / ( totalGeometryaccuracyIntersect.area  + totalGeometryaccuracyDiff.area )
        indexDice = 2 * totalGeometryaccuracyIntersect.area / ( totalGeometryaccuracyIntersect.area  + totalGeometryaccuracyIntersect.area  + totalGeometryaccuracyDiff.area )

        resultsDF.append(
            {
                'Image': image_id,
                'indexJaccard': indexJaccard,
                'indexDice': indexDice,
                'areaObserved':  totalGeometryObserved.area,
                'areaPredicted':  totalGeometryPredicted.area,
                'areaIntersect':  totalGeometryaccuracyIntersect.area,
                'areaDifference':  totalGeometryaccuracyDiff.area,
            }
        )

    resultsDF = DataFrame(resultsDF)
    resultsDF.to_csv(modelDirectory + '/accuracyRaw_' + str(round(threshold, 2)) + '.csv', index = False, header=True)

    resultsDFAverage = pd.concat([resultsDF.mean(axis=0), resultsDF.mean(axis=0)],axis=1, keys=['Mean', 'Stdv'])
    resultsDFAverage.drop(labels='Image',axis=0).to_csv(modelDirectory + '/accuracyAverage_' + str(round(threshold, 2)) + '.csv', index = True, header=True)

    resultsDFThreshold.append(
            {
                'Threshold': threshold,
                'indexJaccard': list(resultsDFAverage['Mean'])[1],
                'indexDice': list(resultsDFAverage['Mean'])[2],
                'areaObserved':  list(resultsDFAverage['Mean'])[3],
                'areaPredicted':  list(resultsDFAverage['Mean'])[4],
                'areaIntersect':  list(resultsDFAverage['Mean'])[5],
                'areaDifference':  list(resultsDFAverage['Mean'])[6],
            }
    )

resultsDFThreshold = DataFrame(resultsDFThreshold)
resultsDFThreshold.to_csv(modelDirectory + '/accuracyThresholdTest.csv', index = False, header=True)
