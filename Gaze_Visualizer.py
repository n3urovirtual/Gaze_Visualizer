import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
import numpy as np


# Draw fixation map
def create_fixation_map(image, fixations, image_extent = [-960, 960, -540, 540]):
    fig, ax = plt.subplots(dpi = 150)

    # Overlay the fixation points on top of the image
    ax.imshow(image, extent = image_extent, aspect='auto')

    for index, row in fixations.iterrows():
        if 'duration' in fixations.columns:
            # Use the fixation duration to determine the size of the marker
            size = row['duration']
        else:
            # Use the specified marker size if no duration is provided
            size = 100
        plt.scatter(row['x'],
                    row['y'],
                    c = "lime",
                    edgecolors = 'black',
                    marker = "o",
                    s = size*10,
                    alpha = 0.5)
    # Return the figure object
    return fig


# Draw scanpath
def create_scanpath(image, scanpath,
                    image_extent = [-960, 960, -540, 540]):
    fig, ax = plt.subplots(dpi = 150)

    # Display the image
    ax.imshow(image, extent = image_extent, aspect = 'auto')

    # Check if the scanpath dataframe has a 'duration' column
    if 'duration' in scanpath.columns:
        # Create a scatterplot of the scanpath, using the durations to
        # set the size and alpha of the markers
        ax.scatter(x = scanpath['x'],
                   y = scanpath['y'],
                   s = scanpath['duration']*10,
                   c = 'lime',
                   edgecolors = 'black')
    else:
        # Create a scatterplot of the scanpath
        ax.scatter(x = scanpath['x'],
                   y = scanpath['y'],
                   c = 'lime',
                   edgecolors = 'black')

    # Connect the fixations with lines
    ax.plot(scanpath['x'],
            scanpath['y'],
            c = 'yellow')

    # Add numbers to each fixation in the scanpath
    for i, (x, y) in enumerate(zip(scanpath['x'], scanpath['y'])):
        ax.text(x,
                y,
                i+1,
                ha = 'center',
                va = 'center')

    # Return the figure object
    return fig


# Draw heatmap
def create_heatmap(image, fixations, image_extent = [-960, 960, -540, 540]):
    fig, ax = plt.subplots(dpi = 150)

    # Display the image
    ax.imshow(image, zorder=0, extent = image_extent, aspect='auto')

    #Create a heatmap using the kernel density estimation plot
    sns.kdeplot(x = fixations['x'],
                y = fixations['y'],
                ax = ax,
                cut = 0,
                bw_method = 0.4,
                bw_adjust = 0.6,
                cbar = False,
                levels = 110,
                shade = True,
                cmap = cm.jet,
                alpha = 0.3)

    # Return the figure object
    return fig