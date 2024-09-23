import pandas as pd
import numpy as np
import os
import cv2
import sys
import math
import matplotlib
import matplotlib.pyplot as plt
from tifffile import imread
from glob import glob
from scipy.stats import mode
from collections import Counter
import math
import skimage
import re
from itertools import chain
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from functools import partial
import matplotlib.animation as animation
import networkx as nx
import skimage
from tifffile import imread
from skimage import morphology


def quick_threshold(img, n_classes, target_classes):
    """A simple function to threshold an image 

    Params:
    ------------
    img (np.array):
        The input image
    n_classes (int):
        number of classes for the multi-ostu thresholding
    target_classes (list of int):
        The class values of interest
        
    Returns:
    ------------
    newImg (np.array):
        Binary thresholded image
    """
    bins = skimage.filters.threshold_multiotsu(image=img, 
                                               classes=n_classes)
    newImg = np.digitize(img, bins=bins)
    newImg = np.where(np.isin(newImg, target_classes), 1, 0)
    return newImg


def get_wound_area(img, foot=skimage.morphology.square, t=0.5, q=2, dilation=[20, 10, 15]):
    """A function explode the cell areas from an un-segmented, processed image

    Params:
    ------------
    img (np.array):
        The input image
    foot (skimage.morphology.[element]):
        A valid structuring element from `skimage.morphology'
    t (float):
        The scale factor for small object removal, assumes circular footprint
    q (float):
        The scale factor for small hole filling, assumes circular footprint
    dialtion (iterable):
        The different size of the footprint used for image dilation

    Returns:
    ------------
    newImg (np.array):
        Binary image with the wound area
    """
    newImg = img.copy()
    
    # dilate the image
    for r in dilation:
        newImg = skimage.morphology.dilation(newImg, selem=foot(r))#footprint=foot(r))

        # remove single blobs
        min_size = t * (math.pi * (r**2))
        max_hole = q * (math.pi * (r**2))
        newImg = newImg.astype(bool)
        newImg = skimage.morphology.remove_small_holes(newImg, max_hole, connectivity=2)
        newImg = skimage.morphology.remove_small_objects(newImg, min_size=min_size)

    newImg = np.invert(newImg)
    return newImg


def filter_cell(nucLabels, data, nn=20, q=0.995):
    """A function to filter segemented cells based on
    their distance to the closest `nn` other cells

    args:
        : nucLabels (np.array): a 2D array with cell labels
        : nn (int): the number of closest cells over which distance
            is averaged
        : q (float): the threshold used to determine if a
            cell is excluded. lowerr threshold excludes more cells

    returns:
        filtered (np.array): a filtered 2D array where cell labels
            that are excluded are dropped
    """
    from sklearn.neighbors import NearestNeighbors

    # building a dataframe of cell positions and predicted probabilities
    dp = pd.DataFrame(data['points'], columns=['y', 'x'])
    dp['prob'] = data['prob']
   
    # get the nearest neighbors to each cell
    nbrs = NearestNeighbors(n_neighbors=nn)
    nbrs.fit(dp[['x', 'y']])
    distances, indices = nbrs.kneighbors(X=dp[['x', 'y']]) # extract the nearest neighbors
   
    # compute the average distance to the `nn` nearest neighbors
    dp['mean_dist'] = np.mean(distances, axis=1)
   
    # threshold points that are "isolated" from nearby cells based on the distribution
    # of distances
    dp['is_isolated'] = dp['mean_dist'] > np.quantile(dp['mean_dist'], q)
   
    # get the index of the cells to keep
    # WARNING labels are the index + 1 !!!!!
    indices_to_drop = dp[dp['is_isolated']].index + 1
   
    # set excluded cells to the background color (zero)
    mask = np.isin(nucLabels, indices_to_drop)
    filtered = nucLabels.copy()
    filtered[mask] = 0
    return filtered, dp


def get_lines_of_best_fit(contours):
    """A function to fit the lines along the contours
    of the wound edge 
    
    args: 
        : contours (list of np.arrays): a list where each element is a 2D np.array of contour
              coordinates for each wound edge. Each row is a point with (x,y) coordinates.
        
    returns:
        : res (list of np.arrays): a list of 2D np.arrays where each array 
              corresponds to the best fit line for a contour 
    
    """
    res = [] # data structure to store the results (no matter how many)

    for i, contour in enumerate(contours):
        x = contour[:, 0].reshape(1, -1).T #x coordinate of contour point
        y = contour[:, 1] #y coordinate of contour point

        reg = LinearRegression().fit(x, y)
        y_hat = reg.predict(x)

        """MAKE A DIFFERENT DATA STRUCTURE"""

        vec = np.vstack([contour[:, 0], y_hat])
        
        res.append(vec.T)

    return res