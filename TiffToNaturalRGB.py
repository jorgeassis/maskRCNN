
## ---------------------------------------------------------------
## ---------------------------------------------------------------
## Resize images

for element in dir():
    if element[0:2] != "__":
        del globals()[element]

del element

import glob
import numpy as np
from osgeo import gdal
#import scipy.misc as sm

import matplotlib.pyplot as plt
import numpy
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
in_dir = '../../Data/Scenes/'
in_dir = "/Volumes/Jellyfish/GDrive/" # '../../Data/Scenes/'

# Where the jpg files are goind to be dumped
dumpFolder = 'Data/jpg/'

listTiffFiles = glob.glob(in_dir + '**.tif')

for i in range(len(listTiffFiles)):   
    
    tileName = listTiffFiles[i].replace(in_dir , "")
    tileName = tileName.replace(".TIF", "")
    tileName = tileName.replace(".tif", "")

    # Open using gdal
    
    ds = gdal.Open(listTiffFiles[i])
    
    red = ds.GetRasterBand(1)
    green = ds.GetRasterBand(2)
    blue = ds.GetRasterBand(3)
    
    red = red.ReadAsArray()
    green = green.ReadAsArray()
    blue = blue.ReadAsArray()
    
    # normalize data to mean +- std
    red = normAvSD(red)
    green = normAvSD(green) 
    blue = normAvSD(blue) 

    # normalize data to 0-1
    # red = normRange(red)
    # green = normRange(green) 
    # blue = normRange(blue) 

    rgb_uint8 = (np.dstack((red,green,blue)) * 255.999).astype(np.uint8)
    
    img_pil = array_to_img(rgb_uint8)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(img_pil)
    plt.show()

    im = Image.fromarray(rgb_uint8)
    im.save(dumpFolder + tileName + ".jpeg")
    
    width, height = im.size   # Get dimensions

    fileNumber = 0
    for w in range(len(list(range(0,width,1024)))-1):
        left = list(range(0,width,1024))[w]+1
        right = list(range(0,width,1024))[w+1]
        
        for h in range(len(list(range(0,height,1024)))-1):
            top = list(range(0,height,1024))[h]+1
            bottom = list(range(0,height,1024))[h+1]
            
            cropped_im = im.crop((left, top, right, bottom))
            # cropped_im.show()
            fileNumber = fileNumber + 1
            cropped_im.save(dumpFolder + tileName + str(fileNumber) + '.jpeg')

