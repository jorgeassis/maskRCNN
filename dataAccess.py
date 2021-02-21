## ---------------------------------------------------------------
## ---------------------------------------------------------------

## conda install -c conda-forge earthengine-api 
## /Users/jorgeassis/miniconda3/bin/python3

## Erase memory objects

for element in dir():
    if element[0:2] != "__":
        del globals()[element]

del element

## Packages and main configuration

from osgeo import gdal
import ee
import geetools
import folium
import geehydro
from datetime import datetime as dt
from IPython.display import display, Image
import urllib.request
from PIL import Image

ee.Authenticate()
ee.Initialize()

## ---------------------------------------------------------------
## ---------------------------------------------------------------
## Get images

## Select collections
## check the catalog: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_SR

landsat = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR").filter(ee.Filter.eq('WRS_PATH', 40)).filter(ee.Filter.eq('WRS_ROW', 37)).select(['B5', 'B3', 'B2'])

## .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',20)) #delete the first hashtag in this line and filter by cloud cover

## Choose dates

landsat = landsat.filterDate('2018-07-15','2018-07-30')
count = landsat.size()
print('Number of scenes to download: ', str(count.getInfo())+'\n')

## ---------------------------------------------------------------
## ---------------------------------------------------------------
## Download images

landsatLeastCloudy = landsat.sort('CLOUD_COVER').first()

parameters = {'min': 0,
              'max': 1000,
              'bands': ['B5', 'B3', 'B2'],
              'dimensions': 512}

geetools.batch.Export.imagecollection.toDrive(landsat, 'LandStatBulkDownload', scale=30)

## ---------------------------------------------------------------
## ---------------------------------------------------------------
## Visualize thumbnail images

thumbnailURL = landsatLeastCloudy.getThumbUrl(parameters)
urllib.request.urlretrieve(thumbnailURL, '../../thumbnail')
im = Image.open('../../thumbnail')
im.show()

## ---------------------------------------------------------------
## ---------------------------------------------------------------
## Resize images

options_list = [
    '-ot Byte',
    '-of JPEG',
    '-b 1',
    '-b 2',
    '-b 3',
    '-scale 0 1500'
]           

options_string = " ".join(options_list)
    
gdal.Translate(
    '../../Data/LC08_040037_20180723.jpg',
    '../../Data/LC08_040037_20180723.tif',
    options=options_string
)
