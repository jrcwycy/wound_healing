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