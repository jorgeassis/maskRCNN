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
from geetools import ui, cloud_mask
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
## check the catalog: https://developers.google.com/earth-engine/datasets/catalog/landsat

## Band designations for the Landsat satellites
## https://developers.google.com/earth-engine/datasets/catalog/landsat

## USGS Landsat catalog examples
## ee.ImageCollection("LANDSAT/LT05/C01/T1_SR")
## ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")

## USGS Landsat 8 metadata
## https://explorer.earthengine.google.com/#detail/LANDSAT%2FLC08%2FC01%2FT1_SR

## ---------------
## ---------------

landsat = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')

## Filter path / row
landsat = landsat.filter(ee.Filter.eq('WRS_PATH', 40)) # 42, 40
landsat = landsat.filter(ee.Filter.eq('WRS_ROW', 37)) # 36, 37

## Filter by point
# point = ee.Geometry.Point(-119.9094, 33.9222)
# landsat = landsat.filterBounds(point)

landsat = landsat.select(['B5', 'B4', 'B3','sr_aerosol','pixel_qa'])
landsat = landsat.filterMetadata('CLOUD_COVER','less_than',10)
print('Number of scenes: ', str(landsat.size().getInfo())+'\n')

landsat = landsat.filterDate('2010-01-01','2020-12-31')
landsat = landsat.filter(ee.Filter.calendarRange(5,9,'month'))
print('Number of scenes: ', str(landsat.size().getInfo())+'\n')

## produce mask for all land pixels
## https://developers.google.com/earth-engine/tutorials/tutorial_api_05
mask = ee.Image('UMD/hansen/global_forest_change_2015')
mask = mask.select('datamask')
mask = mask.eq(2)
mask.getInfo()

collectionSize = landsat.size().getInfo()
landsat = landsat.toList(landsat.size())

for i in range(collectionSize):
    
    image = ee.Image(landsat.get(i))
    image = image.updateMask(mask)

    # https://explorer.earthengine.google.com/#detail/LANDSAT%2FLC08%2FC01%2FT1_SR
    # If the cloud bit (5) is set and the cloud confidence (7) is high
    # or the cloud shadow bit is set (3), then it's a bad pixel.
    # mask = image.select('pixel_qa').bitwiseAnd(1<<5).eq(0)
    # image = image.updateMask(mask)
    
    # Select the image's CFmask band and make all water pixels value 1, all else 0.
    # mask = image.select('pixel_qa').bitwiseAnd(4).eq(0)
    # image = image.updateMask(mask)
    
    # Select the image's CFmask band and make all Terrain Occlusion pixels value 1, all else 0.
    # mask = image.select('pixel_qa').bitwiseAnd(10).eq(0)
    # image = image.updateMask(mask)
    
    image = image.select(['B5', 'B4', 'B3'])
    
    # Visualization
    parameters = {'min': 0,
                  'max': 1000,
                  'bands': ['B3', 'B4', 'B5'],
                  'dimensions': 1024}
    
    thumbnailURL = image.getThumbUrl(parameters)
    urllib.request.urlretrieve(thumbnailURL, '../../thumbnail')
    im = Image.open('../../thumbnail')
    # im.show()

    task = ee.batch.Export.image.toDrive(image = image, folder='LandStatDownload' , fileNamePrefix = image.getInfo()['id'][23:], scale=30)
    task.start()
    # task.status()


## --------------

## Download collection

bands = ['B3', 'B4', 'B5']
geetools.batch.Export.imagecollection.toDrive(landsat, 'LandStatDownload', scale=30)

## ---------------------------------------------------------------
## ---------------------------------------------------------------
