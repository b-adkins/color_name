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

html_dic = collections.OrderedDict(html_dic)
rgb_dic = collections.OrderedDict([(name, colors.colorConverter.to_rgb(spec)) for name, spec in html_dic.iteritems()])
hsv_dic = collections.OrderedDict([(name, colors.rgb_to_hsv(rgb)) for name, rgb in rgb_dic.iteritems()])

hsvs = np.array(hsv_dic.values(), dtype=np.float)
h = hsvs[:, 0]
s = hsvs[:, 1]
v = hsvs[:, 2]

# Colors to classify
in_rgbs = np.array([[26, 189, 107], [171, 39, 101], [84, 98, 131]], dtype=np.float)/255
in_hsvs = [colors.rgb_to_hsv(rgb) for rgb in in_rgbs]

closest_colors = []
closest_names = []
for in_hsv in in_hsvs:
    distances = np.array([np.linalg.norm(hsv - in_hsv) for hsv in hsvs])
    i_closest = np.argmin(distances)
    i_close = np.where(distances < 0.15) 
    closest_colors.append(hsvs[i_closest])
    closest_names.append(hsv_dic.keys()[i_closest])
    
    print in_hsv, 'is most likely', closest_names[-1]
    print 'but could be', np.array(html_dic.keys())[i_close]


# Plot the data set in HSV
fig = pyplot.figure()
ax = fig.add_subplot(221, projection='3d')

hue_angle = h * 2*np.pi
x = s*np.cos(hue_angle)
y = s*np.sin(hue_angle)
pyplot.scatter(x, y, zs=v, c=html_dic.values(), marker=',')

ax = pyplot.subplot(222)
ax.set_title('Hue vs Saturation')
pyplot.scatter(h, s, color=html_dic.values(), marker='.')
for in_hsv, closest_name in zip(in_hsvs, closest_names):
    pyplot.scatter(in_hsv[0], in_hsv[1], c=colors.hsv_to_rgb(in_hsv), marker='p')
    ax.annotate(closest_name, xy=in_hsv[0:2], textcoords='offset points')

ax = pyplot.subplot(224)
ax.set_title('Hue vs Value')
pyplot.scatter(h, v, color=html_dic.values(), marker='.')

pyplot.show()
