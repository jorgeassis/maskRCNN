## --------------------------
## --------------------------

## Erase memory objects

for element in dir():
    if element[0:2] != "__":
        del globals()[element]

del element

## Packages and main functions

import glob
import os
import sys
from osgeo import ogr
import shapefile
from osgeo import gdal

scale = '-scale min_val max_val'
options_list = [
    '-ot Byte',
    '-of JPEG',
    '-b 1',
    '-b 4',
    '-b 3',
    scale
] 
options_string = " ".join(options_list)

def listToString(s): 
    str1 = "" 
    for i in range(len(s)): 
        str1 +=  str(s[i]).replace(".0", "")
        if i != len(s)-1:
            str1 +=  ','
    return str1 

## --------------------------
## --------------------------

## Root directory of the project

rootDirectory = os.path.abspath("/Volumes/Jellyfish/GDrive/Manuscripts/Convolutional Neural Networks for kelp canopy identification/Data/tiledData/LC082210962017041901T1-SC20201001144051")
os.chdir(rootDirectory)
os.getcwd()

fileName = "test.json"

tifDirectory = "TIFS"
shpDirectory = "SHAPEFILES_indexed"
shpFiles = glob.glob(shpDirectory + "/*.shp")

## --------------------------
## --------------------------

len(shpFiles)

## --------------------------
## --------------------------

# Test which have data

tilesWithData = []
tilesWithoutData = []

for shp in range(len(shpFiles)):
    
    shape = shapefile.Reader(shpFiles[shp])
    tileName = shpFiles[shp].replace(shpDirectory + "/", "")
    tileName = tileName.replace(".shp", "")

    # image file
        
    if not os.path.exists(rootDirectory + "/" + tifDirectory + "/" + tileName + ".tif"):
        continue

    if shape.bbox[1] != float("inf"):
        tilesWithData.append(shp)
        
    if shape.bbox[1] == float("inf"):
        tilesWithoutData.append(shp)
        
len(tilesWithData)
len(tilesWithoutData)

## --------------------------
## --------------------------

len(tilesWithData) * 0.7
len(tilesWithoutData) * 0.7

len(tilesWithData) * 0.2
len(tilesWithoutData) * 0.1

trainDataset = tilesWithData[1:530] + tilesWithoutData[1:530]
testDataset = tilesWithData[531:682] + tilesWithoutData[531:682]

dumpFolder = 'train'
listOfFiles = trainDataset

# prepare json file

infoToPopulate = ''

for shp in listOfFiles:
    
    # open shapefile # https://github.com/GeospatialPython/pyshp
    shape = shapefile.Reader(shpFiles[shp])
    n,n,area = shape.records()[0]

    tileName = shpFiles[shp].replace(shpDirectory + "/", "")
    tileName = tileName.replace(".shp", "")

    # image file
        
    if not os.path.exists(rootDirectory + "/" + tifDirectory + "/" + tileName + ".tif"):
        continue

    gdal.Translate(
        rootDirectory + "/" + dumpFolder + "/" + tileName + ".jpg",
        rootDirectory + "/" + tifDirectory + "/" + tileName + ".tif",
        options=options_string
    )
    os.remove(rootDirectory + "/" + dumpFolder + "/" + tileName + ".jpg.aux.xml")
    
    imageSize = os.path.getsize(rootDirectory + "/" + dumpFolder + "/" + tileName + ".jpg") 
    infoToPopulate = infoToPopulate + '{"' + tileName + '":{"filename":"' + tileName + '.jpg","size":' + str(imageSize) + ',"regions":['

    if shape.bbox[1] != float("inf"):
        
        for poly in range(len(shape)):
        
            shapes = shape.shapes()
            coordinates = shapes[poly].points
            
            # feature = shape.shapeRecords()[poly]
            # first = feature.shape.__geo_interface__
            # coordinates = first['coordinates']
                        
            xValues = []
            yValues = []
            
            # for subpoly in range(len(coordinates)):
            #     
            #     subcoordinates = coordinates[subpoly]
            #     
            #     if len(subcoordinates) == 1:
            #         subcoordinates = subcoordinates[0]
            #     
            
            for i in range(len(coordinates)):
                x,y = coordinates[i]
                xValues.append(x)
                yValues.append(y)
            
            xValues = listToString(xValues)
            yValues = listToString(yValues)
                                            
            infoToPopulate = infoToPopulate + '{"shape_attributes":{"name":"polyline","all_points_x":[' + xValues + '],"all_points_y":[' + yValues + ']},"region_attributes":{"region":"kelp","name":"kelp"}}' 
    
            if poly != len(shape)-1: 
                infoToPopulate +=  ',' 
        
    infoToPopulate = infoToPopulate + '],"file_attributes":{}}'
    
    if shp != len(shpFiles)-1: 
        infoToPopulate +=  ',' 

f = open(dumpFolder + '/' + fileName, "w")
f.write(infoToPopulate)
f.close()
