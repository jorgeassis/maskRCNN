

import json
import pandas as pd
import os

# https://www.robots.ox.ac.uk/~vgg/software/via/
# https://www.robots.ox.ac.uk/~vgg/software/via/via.html

annotationFile = '/Volumes/Jellyfish/GDrive/Manuscripts/Convolutional Neural Networks for kelp canopy identification/Git/maskRCNN/Data/train/via_region_data.json'
annotations = json.load(open(annotationFile))
annotations

df = pd.read_json(annotationFile)
df.head()


df.to_json('/Volumes/Jellyfish/GDrive/Manuscripts/Convolutional Neural Networks for kelp canopy identification/Git/maskRCNN/Data/train/via_region_data copy 2.json')
