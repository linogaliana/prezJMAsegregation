import geopandas as gpd
import pandas as pd
import s3fs
import boto3
import py7zr
#import geoplot as gplt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import contextily as ctx
import matplotlib.colors as mc
import os
import matplotlib.patches as mpatches

bucket = "lgaliana"
city = "Paris"
decile = 1
path = "phonesegregation/groupe-787/data/ raw_spark_output/" + city + "_D" + str(decile) + "_deviationHour.csv"
paper_directory = "../data"

fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': "https://minio.lab.sspcloud.fr"})
s3 = boto3.client('s3', endpoint_url= "https://minio.lab.sspcloud.fr")

if city=='Marseille':
    cols = ['day', 'hour', 'simulation', 
                            'class', 'period',
                            'p_grid', 'grid_id', 'p_pop','lastcol']
else:
    cols = ['day', 'hour', 'simulation',
                            'class', 'period',
                            'p_grid', 'grid_id', 'p_pop']

with fs.open('s3://' + bucket + "/" + path) as f:
    df = pd.read_csv(f,
                     names=cols)
    
df = df[(df['simulation'] == 'simulation_0') & (df['day'] == "weekjob")]
df = df[df['hour'].isin([6, 16, 20, 23])]

df2 = (df
    .assign(
    sum_p_grid=lambda x: x.groupby(['grid_id', 'hour']).transform('sum')['p_grid'])
)

df3 = df2[(df2['class'] == 1) | ((df2['class'] == 0) & (df2['p_grid'] == df2['sum_p_grid']))]

df3.loc[df3['class'] == 0, 'p_grid'] = 0
df3 = df3[df3['sum_p_grid'] > 0]


s3.download_file(bucket,
                 "phonesegregation/groupe-787/data/grid_shp_france/grid_shp_france.7z",
                 'grid_shp_france.7z')

archive = py7zr.SevenZipFile('grid_shp_france.7z', mode='r')
archive.extractall(path= paper_directory + "/grid")
archive.close()

shapefile = gpd.read_file("%s/grid/grid_shp_france/grid_500_france.shp" % paper_directory)


def shapefile_hour(data, hour, shapefile, colname='p'):
    """
    Generic function to match shapefile with dataframe
    :param data: Dataframe that should be matched
    :param hour: Hour to consider
    :param shapefile: Cell level shapefile
    :param colname: Column name
    :return:
    A Geopandas object that matches data and geometry from
      shapefile
    """
    datacopy = data[(data['hour'] == hour)]
    datacopy = datacopy.assign(prop=100 * datacopy[colname] / datacopy['sum_p_grid'])
    datacopy['prop'] = pd.to_numeric(datacopy['prop'], errors = "coerce")
    return shapefile.merge(datacopy, on="grid_id")

shp_6b = shapefile_hour(data = df3, hour = 6, shapefile = shapefile, colname="p_grid")
shp_16b = shapefile_hour(data = df3, hour = 16, shapefile = shapefile, colname="p_grid")
shp_20b = shapefile_hour(data = df3, hour = 20, shapefile = shapefile, colname="p_grid")
shp_23b = shapefile_hour(data = df3, hour = 23, shapefile = shapefile, colname="p_grid")

# TO WRITE FILE -----------------------------------------------------

# shp_6b.to_file("hour6_" + city + ".geojson", driver='GeoJSON')
# shp_16b.to_file("hour16_" + city + ".geojson", driver='GeoJSON')
# s3 = boto3.resource('s3', endpoint_url='http://minio.stable.innovation.insee.eu/')
# s3.Bucket("groupe-787").upload_file("hour6_" + city + ".geojson", "data/donnees_IA/" + "hour6_" + city + "_D" + str(decile) + ".geojson")
# s3.Bucket("groupe-787").upload_file("hour16_" + city + ".geojson", "data/donnees_IA/" + "hour16_" + city + "_D" + str(decile) + ".geojson")
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(ncols = 2, figsize = (20,20))
# shp_6b.plot('prop', cmap = "Reds", scheme ="User_Defined", classification_kwds = dict(bins=[3,6,9,12,100]), ax = axes[0], legend = True, legend_kwds = {"loc": "lower left"})
# shp_16b.plot('prop', cmap = "Reds", scheme ="User_Defined", classification_kwds = dict(bins=[3,6,9,12,100]), ax = axes[1], legend = True, legend_kwds = {"loc": "lower left"})

# IMPORT CITY LEVEL LIMITS ---------

xmin = 2.10285
xmax = 2.53946
ymax = 48.996438
ymin = 48.757567

s3.download_file(bucket, "phonesegregation/groupe-787/data/shapefiles/communes/comf.TAB",
                 "comf.TAB")
s3.download_file(bucket, "phonesegregation/groupe-787/data/shapefiles/communes/comf.MAP",
                 "comf.MAP")
s3.download_file(bucket, "phonesegregation/groupe-787/data/shapefiles/communes/comf.DAT",
                 "comf.DAT")
s3.download_file(bucket, "phonesegregation/groupe-787/data/shapefiles/communes/comf.IND",
                 "comf.IND")
s3.download_file(bucket, "phonesegregation/groupe-787/data/shapefiles/communes/comf.ID",
                 "comf.ID")

shp_com = gpd.read_file("comf.TAB", driver="MapInfo File", crs=27572)
shp_com = shp_com.to_crs(4326)