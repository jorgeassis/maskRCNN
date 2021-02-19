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

/Users/jorgeassis/miniconda3/bin/python3

## Erase memory objects

for element in dir():
    if element[0:2] != "__":
        del globals()[element]

del element

## Packages and main configuration

import os
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

import math
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
datasetDirectory = "../../Data/4200/"

## Local path to trained weights file
weightsFilePath = os.path.join("../../", "mask_rcnn_coco.h5")

class mainConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    #
    # Details: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py
    #
    # Set the limits of the image shape. Use small images for faster training. 
    #IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 1024
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3
    # Give the configuration a recognizable name
    NAME = "kelp" # trash
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + shapes (4)
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    N_EPOCHS = 1 # ??

config = mainConfig()
config.display()

## ------------------------------------------------------------------------
## ------------------------------------------------------------------------

## Dataset

class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "kelp") # check name in via_region_data.json
        # self.add_class("object", 2, "clouds")
        # self.add_class("object", 3, "paper")
        # self.add_class("object", 4, "trash")
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        # We mostly care about the x and y coordinates of each region
        annotations1 = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        # print(annotations1)
        annotations = list(annotations1.values())  # don't need the dict keys
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['name'] for s in a['regions']]
            print("objects:",objects)
            # name_dict = {"bottle": 1,"glass": 2,"paper": 3,"trash": 4}
            name_dict = {"kelp": 1}
            # key = tuple(name_dict)
            num_ids = [name_dict[a] for a in objects]
            # num_ids = [int(n['Event']) for n in objects]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            self.add_image(
                "object",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a bottle dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
        	rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        	mask[rr, cc, i] = 1
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

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
dataset_val.load_custom(args.dataset, "val")
dataset_val.prepare()

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

config.LEARNING_RATE

t0 = time.time()

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=config.N_EPOCHS,
            layers='heads')

t1 = time.time()

t2 = time.time()

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2, 
            layers="all")

t3 = time.time()

t1-t0
t3-t2

## Save weights

# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model.keras_model.save_weights(weightsFilePath)

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

# Get path to saved weights
# Either set a specific path or find last trained weights

# weightsFilePathFinal = weightsFilePath
# weightsFilePathFinal = model.find_last() # Check https://github.com/matterport/Mask_RCNN/issues/885
# weightsFilePathFinal = "../../logs/kelp20210219T1027/mask_rcnn_kelp_0001.h5"

import glob

files = glob.glob("../../logs/" + '/**/*.h5', recursive=True)
import os
>>> statinfo = os.stat('somefile.txt')
>>> statinfo
statinfo.st_mtime

# Load trained weights

model.load_weights(weightsFilePathFinal, by_name=True)

## --------------------------
## --------------------------
# Test on a val image

image_id = random.choice(dataset_val.image_ids)
image_id = 0

original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inferenceConfig, 
                           image_id, use_mini_mask=False)

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
boundaryO.plot(color = 'red')
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

index = totalGeometryaccuracyUnion / ( totalGeometryObserved.area + totalGeometryPredicted.area - totalGeometryaccuracyUnion )

# Dice Coefficient

index = 2 * totalGeometryaccuracyUnion / ( 2 *  totalGeometryaccuracyUnion + ( totalGeometryObserved.area + totalGeometryPredicted.area - totalGeometryaccuracyUnion ))

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

externalImagePath = "Data/0 Bigger size.png" # Data/val/3.png # Data/0.png # Data/0 Bigger size.png

# display image

img = mpimg.imread(externalImagePath)
imgplot = plt.imshow(img)
plt.show()

img = load_img(externalImagePath)
img = img_to_array(img)

results = model.detect([img], verbose=1)
r = results[0]

visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], figsize=(8, 8))

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
