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


shp_6b = shapefile_hour(data = df3, hour = 6, shapefile = shapefile, colname="p_grid")
shp_16b = shapefile_hour(data = df3, hour = 16, shapefile = shapefile, colname="p_grid")
shp_20b = shapefile_hour(data = df3, hour = 20, shapefile = shapefile, colname="p_grid")
shp_23b = shapefile_hour(data = df3, hour = 23, shapefile = shapefile, colname="p_grid")


def expand_points(shapefile,
                  index_var = "grid_id",
                  weight_var = 'prop',
                  radius_sd = 100,
                  crs = 2154):
    """
    Multiply number of points to be able to have a weighted heatmap
    :param shapefile: Shapefile to consider
    :param index_var: Variable name to set index
    :param weight_var: Variable that should be used
    :param radius_sd: Standard deviation for the radius of the jitter
    :param crs: Projection system that should be used. Recommended option
      is Lambert 93 because points will be jitterized using meters
    :return:
      A geopandas point object with as many points by index as weight
    """

    shpcopy = shapefile
    shpcopy = shpcopy.set_index(index_var)
    shpcopy['npoints'] = np.ceil(shpcopy[weight_var])
    shpcopy['geometry'] = shpcopy['geometry'].centroid
    shpcopy['x'] = shpcopy.geometry.x
    shpcopy['y'] = shpcopy.geometry.y
    shpcopy = shpcopy.to_crs(crs)
    shpcopy = shpcopy.loc[np.repeat(shpcopy.index.values, shpcopy.npoints)]
    shpcopy['x'] = shpcopy['x'] + np.random.normal(0, radius_sd, shpcopy.shape[0])
    shpcopy['y'] = shpcopy['y'] + np.random.normal(0, radius_sd, shpcopy.shape[0])

    gdf = gpd.GeoDataFrame(
        shpcopy,
        geometry = gpd.points_from_xy(shpcopy.x, shpcopy.y),
        crs = crs)

    return gdf

cols = ['day', 'hour', 'simulation',
                            'class', 'period',
                            'p_grid', 'grid_id', 'p_pop']

with fs.open('s3://' + bucket + "/" + path) as f:
    df = pd.read_csv(f,
                     names=cols)
    
df = df[(df['simulation'] == 'simulation_0') & (df['day'] == "weekjob")]

def heatmap_hour(data, hour = 6):
    df2 = data[data['hour'] == hour]

    df2 = (df2
        .assign(
        sum_p_grid=lambda x: x.groupby(['grid_id', 'hour']).transform('sum')['p_grid'])
    )

    df3 = df2[(df2['class'] == 1) | ((df2['class'] == 0) & (df2['p_grid'] == df2['sum_p_grid']))]

    df3.loc[df3['class'] == 0, 'p_grid'] = 0
    df3 = df3[df3['sum_p_grid'] > 0]    
    shp_6b = shapefile_hour(data = df3, hour = hour, shapefile = shapefile, colname="p_grid")
    shp_6b['x'] = shp_6b.geometry.centroid.x
    shp_6b['y'] = shp_6b.geometry.centroid.y

    df = pd.DataFrame(shp_6b.drop(columns='geometry'))
    df = df[["x","y","prop"]].dropna()
    df2 = btbpy.kernelSmoothing(df, "2154", 500, 2000)

    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize = (10,10))

    df2.to_crs(4326).plot(ax=ax,
                          column = "prop",
                          edgecolor=None,
             cmap = matplotlib.colors.ListedColormap(['#1a9641', '#a6d96a', '#ffffc0', '#fdae61', '#d7191c']),
             norm = matplotlib.colors.BoundaryNorm(vals, cmap.N),
                          alpha = 0.6,
             figsize=(12,10))
    ctx.add_basemap(ax = ax, source=ctx.providers.Stamen.Toner, crs = 4326)
    plt.axis('off')
    ax.set_xlim(2.10285, 2.53946)
    ax.set_ylim(48.757567, 48.996438)
    plt.savefig('output/hour' + str(hour) + ".png")
    
    
# [heatmap_hour(df, i) for i in range(24)]

import glob
x = glob.glob("./output/*.png")
import re

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

x.sort(key=natural_keys)
x

import imageio
images = []
for filename in x:
    images.append(imageio.imread(filename))
imageio.mimsave('./output/movie.gif', images)