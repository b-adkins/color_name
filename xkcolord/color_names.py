#!/usr/bin/env python

import rospy as ros
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
from xkcolord.srv import *

# Kludge to include a file a few two directories up
import sys
import os
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(lib_path)
sys.path.append(lib_path)
from centroid_classifier import ColorNamer

class ROSColorNamer(ColorNamer):
    def color_centroid(self, msg):
        return super(ColorNamer, self).color_centroid(msg.label)

def main():
    ros.init_node('xkcolord', anonymous=True)

    # @todo Refactor to param server
    color_names_file = './rgb_names.yaml'
    color_namer = ROSColorNamer(color_names_file)

    ros.Service('color_centroid', color_centroid, color_namer.color_centroid)
#    ros.Service('is_color', color_namer.is_color)

if __name__ == '__main__':
    main()
