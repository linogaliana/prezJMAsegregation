#!pip install -r requirements.txt

import boto3
import s3fs
import py7zr
import numpy as np
import pandas as pd
import geopandas as gpd
#import geoplot as gplt
from shapely.geometry import LineString
import contextily as ctx
import mapclassify as mc
import matplotlib.pyplot as plt
import os


bucket = "lgaliana"
paper_directory = "../data"
s3 = boto3.client('s3', endpoint_url= "https://minio.lab.sspcloud.fr")
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': "https://minio.lab.sspcloud.fr"})

path_data_gravity = "phonesegregation/groupe-787/data/ raw_spark_output"

# SELECTION CELLS IN A CITY -------------------

# 1/ IMPORT CELL LEVEL SHAPEFILE ===============

s3.download_file(bucket,
                 "phonesegregation/groupe-787/data/grid_shp_france/grid_shp_france.7z",
                 'grid_shp_france.7z')

archive = py7zr.SevenZipFile('grid_shp_france.7z', mode='r')
archive.extractall(path= paper_directory + "/grid")
archive.close()

shapefile = gpd.read_file("%s/grid/grid_shp_france/grid_500_france.shp" % paper_directory)

# 2/ IMPORT CITY NAMES SHAPEFILE ===============

for extension in ["TAB", "MAP", "DAT", "IND", "ID"]:
    s3.download_file(bucket, "phonesegregation/groupe-787/data/shapefiles/communes/comf." + extension,
                     "comf." + extension)

shp_com = gpd.read_file("comf.TAB", driver="MapInfo File", crs=27572)
shp_com = shp_com.to_crs(4326)