import numpy as np
from skimage import io
import skimage
import os
import scipy.misc
from skimage import data
from scipy import ndimage
from skimage.color import rgb2gray, gray2rgb, rgb2hsv
import os
import numpy as np
import pandas as pd
import win32api
import scipy.special
import seaborn as sns


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

painting_categories = ['ceiling painting', 'oil sketch', 'painting', 'print', 'triptych', 'watercolor (painting)']
drawing_categories = ['drawing', 'ethnographic', 'graphic design (guide term)', 'sketchbook, drawing']

"""
create_dataframe

Output : Scans all image files in our database of images and creates a dataframe by generating image stats
"""

def create_dataframe():
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


