import collections

import matplotlib.colors as colors
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.linalg
import pandas as pd
import yaml

# Load XKCD color labels
with open('./rgb_names.yaml') as f:
    html_dic = yaml.load(f)  # maps names to hexidecimal RGB, e.g. "#xxxxxx"

names = html_dic.keys()
specs = html_dic.values()
rgbs = np.array([colors.colorConverter.to_rgb(spec) for spec in specs], dtype=np.float)
hsvs = np.array([colors.rgb_to_hsv(rgb) for rgb in rgbs], dtype=np.float)

df = pd.DataFrame(np.hstack([np.array(zip(names, specs), dtype=np.string_), rgbs, hsvs]))
#                  dtype=[np.object, np.object, np.float, np.float, np.float, np.float, np.float, np.float])
df.columns = ['name', 'html', 'r', 'g', 'b', 'h', 's', 'v']
df.index = df['name']

# Colors to classify


def match_color_label(color, colorspace='rgb', label='', threshold=0.15):
    '''
    color A length 3 array-like of RGB or HSV values ranging from 0.0 to 1.0.
    colorspace RGB or HSV.
    label A color name, e.g. 'teal'.
    threshold Maximum Euclidean distance to label centroid in HSV space to be considered a match.
    return True or False
    '''
    if colorspace == 'rgb':
        label_centroid = df[['r', 'g', 'b']][label].as_float()
    elif colorspace == 'hsv':
        label_centroid = df[['h', 's', 'v']][label].as_float()
    else:
        raise ValueError('Unrecognized colorspace "". Must be "rgb" or "hsv".'.format(colorspace))
       
    distance = np.linalg.norm(np.array(color, dtype=np.float) - label_centroid)
    return distance <= threshold


def classify_color(color, colorspace='rgb'):
    '''
    return (name, distance) Distance in [0.0, sqrt(3)].
    '''
    if colorspace == 'rgb':
        color = colors.rgb_to_hsv(color)
    elif colorspace != 'hsv':
        raise ValueError('Unrecognized colorspace "". Must be "rgb" or "hsv".'.format(colorspace))

    distances = np.array([np.linalg.norm(hsv - color) for hsv in hsvs])
    
    i_closest = np.argmin(distances)
    
    return names[i_closest], distances[i_closest]


if __name__ == '__main__':
    classify_me_rgbs = np.array([[26, 189, 107], [171, 39, 101], [84, 98, 131]], dtype=np.float)/255
    for color in classify_me_rgbs:
        print color, 'is', classify_color(color)[0]

    
# Plot the candidate colors in HSV
# fig = pyplot.figure()
# ax = fig.add_subplot(221, projection='3d')

# hue_angle = h * 2*np.pi
# x = s*np.cos(hue_angle)
# y = s*np.sin(hue_angle)
# pyplot.scatter(x, y, zs=v, c=html_dic.values(), marker=',')

# ax = pyplot.subplot(222)
# ax.set_title('Hue vs Saturation')
# pyplot.scatter(h, s, color=html_dic.values(), marker='.')
# for in_hsv, closest_name in zip(in_hsvs, closest_names):
    # pyplot.scatter(in_hsv[0], in_hsv[1], c=colors.hsv_to_rgb(in_hsv), marker='p')
    # ax.annotate(closest_name, xy=in_hsv[0:2], textcoords='offset points')

# ax = pyplot.subplot(224)
# ax.set_title('Hue vs Value')
# pyplot.scatter(h, v, color=html_dic.values(), marker='.')

# pyplot.show()
