# @todo Refactor into class, have an HSV and an RGB object

import collections

import matplotlib.colors as colors
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.linalg
import pandas as pd
import yaml

class ColorNamer():
    def __init__(self, name_file):
        self.path = name_file
        
        with open(self.path) as f: 
            html_dic = yaml.load(f) # maps names to hexidecimal RGB, e.g. "#xxxxxx"

        self.names = html_dic.keys()
        specs = html_dic.values()
        self.colors = np.array([colors.colorConverter.to_rgb(spec) for spec in specs], dtype=np.float)
        self.name_map = {n: c for n, c in zip(self.names, self.colors)}

    def color_centroid(self, label):
        """
        Returns color most representative of a color name.
    
        :param label: Color name, e.g. 'fuchsia'.
        :return: length 3 color value
        """
        return self.name_map[label]

    def is_color(self, color, label, threshold=0.15):
        """
        Whether this color matches this label.
    
        :param color: A length 3 array-like of RGB values ranging from 0.0 to 1.0.
        :param label: A color name, e.g. 'teal'
        :param threshold: Maximum Euclidean distance to label centroid in HSV space to be considered a match.
        :returns: True or False
        """
        label_centroid = self.color_centroid(label)
        distance = np.linalg.norm(np.array(color, dtype=np.float) - label_centroid)
        return distance <= threshold

    def classify_color(self, color):
        """
        Finds name that best matches this color.
        
        :param color: A length 3 array-like of RGB values ranging from 0.0 to 1.0.
        :returns: (name, distance) Distance between 0 and sqrt(3).
        """      
        distances = np.linalg.norm(np.array(color, dtype=np.float) - self.colors, axis=1)
        i_closest = np.argmin(distances)       
        return self.names[i_closest], distances[i_closest]

if __name__ == '__main__':
    classify_me_rgbs = np.array([[26, 189, 107], [171, 39, 101], [84, 98, 131]], dtype=np.float)/255
    color_namer = ColorNamer('./rgb_names.yaml')

    for color in classify_me_rgbs:
        print color, 'is', color_namer.classify_color(color)[0]

# Test centroid classifier

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
