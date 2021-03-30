
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

# Where the landsat files are
in_dir = '../../Data/Scenes/'

# Where the jpg files are goind to be dumped
dumpFolder = 'Data/jpg/'

b1_file = glob.glob(in_dir + '**B5.TIF') # near infra-red band
b2_file = glob.glob(in_dir + '**B3.TIF') # green band
b3_file = glob.glob(in_dir + '**B2.TIF') # blue band

def normAvSD(band):
    band_mean = band[band != 0].mean()
    band_Std = band[band != 0].std()
    band_min = band_mean - band_Std
    band_max = band_mean + band_Std
    band[band >= band_max] = band_max
    band[band <= band_min] = band_min
    #return(exposure.rescale_intensity(band, in_range=(band_min,band_max)))
    return ((band - band_min)/(band_max - band_min))

for i in range(len(b1_file)):   
    
    tileName = b1_file[i].replace(in_dir , "")
    tileName = tileName.replace(".TIF", "")
    tileName = tileName[:-3]
    
    # Open each band using gdal
    b1_link = gdal.Open(b1_file[i])
    b2_link = gdal.Open(b2_file[i])
    b3_link = gdal.Open(b3_file[i])
    
    b1 = b1_link.ReadAsArray()
    b2 = b2_link.ReadAsArray()
    b3 = b3_link.ReadAsArray()

    # b1 = skimage.exposure.equalize_hist(b1)
    # b2 = skimage.exposure.equalize_hist(b2)
    # b3 = skimage.exposure.equalize_hist(b3)

    # normalize data to mean +- std
    
    b1 = normAvSD(b1)
    b2 = normAvSD(b2) 
    b3 = normAvSD(b3) 

    # create three color image
    rgb = np.dstack((b2, b1, b3))
    #numpy.amax(rgb)
    #numpy.average(rgb)
    #rgb.shape

    # Visualize RGB
    
    img_pil = array_to_img(rgb)
    
    #fig = plt.figure()
    #ax = fig.add_subplot()
    #ax.imshow(img_pil)
    #plt.show()

    img_array = img_to_array(img_pil)
    img_array = img_array.astype(np.uint8)
    
    #img_array = img_array * 255
    
    #numpy.amax(img_array)
    #numpy.amin(img_array)
    #numpy.average(img_array)
    
    im = Image.fromarray(img_array)
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

