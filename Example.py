import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Gaze_Visualizer import create_fixation_map, create_heatmap, create_scanpath

# Load the image and the fixations
image_fix = plt.imread(r'Images/square.JPG')
fixations = pd.DataFrame({'x': [-50, 200, -320, 550, -250],
                          'y': [-50, 0, 200, 270, 380],
                          'duration': [100, 200, 50, 150, 75]})
# Visualize the fixation map on the image
fix_map_figure = create_fixation_map(image_fix, fixations)


# Load the image and the scanpath
image_scan = plt.imread(r'Images/nature.jpg')
scanpath = pd.DataFrame({'x': [0, 200, -720, -800, -500, 500],
                          'y': [0, 340, 0, -50, -400, -130],
                          'duration': [100, 200, 50, 65, 150, 80]})
# Visualize the scanpath on the image
scanpath_figure = create_scanpath(image_scan, scanpath)


# Load the image and the fixations
image_heat = plt.imread(r'Images/sports_website.png')
fixations = pd.DataFrame({'x': np.random.randint(-960,960,size=30),
                          'y': np.random.randint(-540,540,size=30)})
# Visualize the heatmap on the image
heatmap_figure = create_heatmap(image_heat, fixations)
