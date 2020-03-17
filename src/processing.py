import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage
import os
import scipy.misc
from skimage import data
from scipy import ndimage
from skimage.color import rgb2gray, gray2rgb, rgb2hsv
import os
import numpy as np
import bokeh
from bokeh.io import output_notebook, curdoc
from bokeh.plotting import ColumnDataSource
from bokeh.plotting import figure, output_file, show

import bokeh.models as bmo
import win32api
import scipy.special
import seaborn as sns
import os
import umap
import pandas as pd

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback


def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    # global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


tfback._get_available_gpus = _get_available_gpus
from keras import backend as K

K.set_image_data_format('channels_first')



"""
grab_stats

Input : Image Data
Output : Mean Saturation, Mean Brightness, Per Row Variance, Per Row Average, Pixel Count, Resolution as string, Mean
         Hue and Color Complexity
"""

def grab_stats(image):
    try:
        if len(image.shape) == 2:
            image = gray2rgb(image)
        hsv_img = rgb2hsv(image)
        saturation_img = hsv_img[:, :, 1]
        value_img = hsv_img[:, :, 2]
        hue_img = hsv_img[:, :, 0]
        pixelcount = image.shape[0] * image.shape[1]
        mean_saturation = np.mean(saturation_img, axis=(0, 1))
        mean_brightness = np.mean(value_img)
        mean_hue = np.mean(hue_img, axis=(0, 1))
        per_row_var = np.var(value_img, axis=1)
        per_row_avg = np.mean(per_row_var)
        resolution = '{0} x {1}'.format(image.shape[0], image.shape[1])
        unique_colors = np.unique(image.reshape(-1, image.shape[-1]), axis=0).shape[0]
        return mean_saturation, mean_brightness, per_row_var, per_row_avg, pixelcount, resolution, mean_hue, unique_colors
    except ValueError:
        return None

"""
edge_score_calculate

Input : Image Data
Output : Edge Score of image
"""

def edge_score_calculate(img):
    if len(img.shape) == 2:
        img = gray2rgb(img)
    hsv_img = rgb2hsv(img)
    value_img = hsv_img[:, :, 2]

    sobel_x = ndimage.sobel(value_img, axis=0, mode='constant')
    sobel_y = ndimage.sobel(value_img, axis=1, mode='constant')
    edge_image = np.hypot(sobel_x, sobel_y)
    edge_score = np.sum(edge_image)
    return edge_score

"""
List of categories for paintings and drawings
"""

painting_categories = ['oil sketch', 'painting', 'watercolor (painting)']
drawing_categories = ['drawing', 'sketchbook, drawing']

"""
create_dataframe

Output : Scans all image files in our database of images and creates a dataframe by generating image stats
"""

def create_dataframe(**cfg):

    if cfg["load_dataset" ] == 1:
        return pd.read_csv('test_df.csv')

    else:

        df_image_stats = pd.DataFrame(
            columns=['Mean Saturation', 'Mean Brightness', 'Mean Hue', 'Edge Score', 'Per Row Average Variance',
                    'Pixel Count', 'Image Directory', 'Date', 'Description', 'Resolution', 'Category', 'Object Type',
                    'Color Complexity'])

        main_dir = 'images'
        for obj_type in os.listdir(main_dir):
            if (obj_type in painting_categories) or (obj_type in drawing_categories):
                for year in os.listdir(main_dir + '\\' + obj_type):
                    files = os.listdir(main_dir + '\\' + obj_type + '\\' + year)
                    for file in files:
                        directory_file = (main_dir + '\\' + obj_type + '\\' + year + '\\' + file)
                        img = io.imread(directory_file)
                        stats = grab_stats(img)
                        if not stats == None:
                            if obj_type in painting_categories:
                                category = 'Painting'
                            else:
                                category = 'Drawing'
                            edge_score = edge_score_calculate(img)
                            directory = main_dir + '\\' + obj_type + '\\' + year + '\\' + file
                            description = 'Name : {0}, Year : {1}, Object Type: {2}'.format(file, year, obj_type)
                            df_image_stats = df_image_stats.append(
                                {'Mean Saturation': stats[0], 'Mean Brightness': stats[1], 'Mean Hue': stats[6],
                                'Edge Score': edge_score, 'Per Row Average Variance': stats[3], 'Pixel Count': stats[4],
                                'Resolution': stats[5],
                                'Image Directory': directory, 'Date': year, 'Description': description,
                                'Category': category, 'Object Type': obj_type, 'Color Complexity': stats[7]},
                                ignore_index=True)
        # Cleans data by removing missing dates rows as well as those with a low pixel count
        df_image_stats = df_image_stats[df_image_stats['Date'] != 'Date Missing']
        df_image_stats['Date'] = df_image_stats['Date'].astype(int)
        df_image_stats = df_image_stats[df_image_stats['Pixel Count'] > 200000]
        return df_image_stats
"""
generate_stats

Input : DataFrame of image stats
Output : Prints out the statistics of the dataframe
"""

def generate_stats(df):


    #Calculate color complexity
    df['Color Complexity'] = df['Color Complexity'].apply(lambda x: x / (255 ** 3))
    # Normalize edge score and per row variance
    max_edgescore = max(df['Edge Score'])
    df['Edge Score'] = df['Edge Score'].apply(lambda x: x / max_edgescore)

    max_var = max(df['Per Row Average Variance'])
    df['Per Row Average Variance'] = df['Per Row Average Variance'].apply(lambda x: x / max_var)
    # Calculate complexity score
    df['Complexity Score'] = (df['Edge Score'] + df['Per Row Average Variance'] + df['Color Complexity']) / 3
    print(df.describe())

    # Create complexity score graph
    output_file("complexity.html")

    dates = df['Date'].values
    comp_score = df['Complexity Score'].values
    img_dir = df['Image Directory'].values

    p = figure(x_range=(min(dates) - 5, max(dates) + 5), y_range=(0, 1), title="Complexity Graph")
    p.image_url(url=img_dir, x=dates, y=comp_score, w=1, h=0.01)

    # Use keras and Tensorflow to create feature vectors of data

    model = VGG16(weights='imagenet', include_top=False)

    image_data = []
    years = []
    main_dir = 'images'
    for obj_type in os.listdir(main_dir):
        if (obj_type in painting_categories) or (obj_type in drawing_categories):
            for year in os.listdir(main_dir + '\\' + obj_type):
                if year != 'Date Missing':
                    files = os.listdir(main_dir + '\\' + obj_type + '\\' + year)
                    for file in files:
                        years.append(int(year))
                        directory_file = (main_dir + '\\' + obj_type + '\\' + year + '\\' + file)
                        img = load_img(directory_file, target_size=(224, 224))
                        img = img_to_array(img)
                        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
                        img = preprocess_input(img)
                        vgg16_feature = model.predict(img)
                        vgg16_feature_np = np.array(vgg16_feature)
                        vgg16_feature_vector = vgg16_feature_np.flatten()
                        image_data.append(vgg16_feature_vector)

    reducer = umap.UMAP(random_state=42)
    reducer.fit(image_data)
    embedding = reducer.transform(image_data)

    # Generate UMAP plot

    plt.figure(figsize=(10, 10), dpi=80)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=years, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar().set_ticks(list(set(years)))
    plt.title('UMAP projection of Mondrian\'s paintings dataset', fontsize=24);
    plt.savefig('umap_plot.png')

