
## ---------------------------------------------------------------
## ---------------------------------------------------------------
## Resize images

for element in dir():
    if element[0:2] != "__":
        del globals()[element]

del element
import os

import glob
import numpy as np
from osgeo import gdal
#import scipy.misc as sm

import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

from PIL import Image

def normAvSD(band):
    band_mean = np.nanmean(band[band != 0])
    band_Std = np.nanstd(band[band != 0])
    band_min = band_mean - band_Std
    band_max = band_mean + band_Std
    band[band >= band_max] = band_max
    band[band <= band_min] = band_min
    return ((band - band_min)/(band_max - band_min))

def normRange(band):
    band_min = np.nanmin(band)
    band_max = np.nanmax(band)
    return ((band - band_min)/(band_max - band_min))

# Where the landsat files are
in_dir = '/Volumes/Mask RCNN Kelp/Raw Data/Landsat5 (2003-2013)/'

# Where the jpg files are goind to be dumped
dumpFolder = '/Volumes/Jellyfish/GDrive/Manuscripts/Convolutional Neural Networks for kelp canopy identification/Data/Dataset 5/Unsorted/'
dumpFolder = '/Volumes/Mask RCNN Kelp/Annotations Data/Unsorted/'

listTiffFiles = glob.glob(in_dir + '**.tif')

# gdal parameters
options_list = [
    '-ot Byte',
    '-of JPEG',
    '-b 1',
    '-b 2',
    '-b 3',
    '-scale 0 700'
]           
options_string = " ".join(options_list)
    
produceTiles = True # True False

for i in range(len(listTiffFiles)):   
    
    tileName = listTiffFiles[i].replace(in_dir , "")
    tileName = tileName.replace(".TIF", "")
    tileName = tileName.replace(".tif", "")
    
    gdal.Translate(
        listTiffFiles[i].replace(".tif", ".jpg"),
        listTiffFiles[i],
        options=options_string
    )
    os.remove(listTiffFiles[i].replace(".tif", ".jpg.aux.xml"))

    # Get dimensions
    ds = gdal.Open(listTiffFiles[i].replace(".tif", ".jpg"))
    width = ds.RasterXSize
    height = ds.RasterYSize 

    if produceTiles:
    
        fileNumber = 0
        for w in range(len(list(range(0,width,1024)))-1):
            left = list(range(0,width,1024))[w]+1
            right = list(range(0,width,1024))[w+1]+1
            
            for h in range(len(list(range(0,height,1024)))-1):
                top = list(range(0,height,1024))[h]+1
                bottom = list(range(0,height,1024))[h+1]+1
                
                fileNumber = fileNumber + 1
    
                dumpFileName = listTiffFiles[i].replace(".tif", '_' + str(fileNumber) + ".jpg")
                dumpFileName = dumpFileName.replace(in_dir, dumpFolder)
                gdal.Translate(
                                dumpFileName,
                                listTiffFiles[i].replace(".tif", ".jpg"),
                                projWin = [left, top , right , bottom] )
                                

