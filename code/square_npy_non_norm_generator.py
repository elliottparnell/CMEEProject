#!/usr/bin/env python3

""" Reads in images an performs preprocessing making tem ready for CNN analysis"""

__appname__ = 'square_cnn_generator.py'
__author__ = '[Elliott Parnell (ejp122@ic.ac.uk)]'
__version__ = '0.0.1'


import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import numpy.ma as ma
from PIL import Image
import rawpy
import sys
import torchvision.transforms as transforms
import torch 
import pickle
#import tensorflow as tf

def read_raw_img(filename):
    """Reads a raw image format and outputs post proccessing image"""
    try:
        with rawpy.imread(filename) as raw:
            processed_img = raw.postprocess()
        return processed_img
    except:
        print("Please provide raw image format e.g. CR2")
        return 
    
def smallestbox(a,img):
    r = a.any(1)
    if r.any():
        m,n = a.shape
        c = a.any(0)
        out = img[r.argmax():m-r[::-1].argmax(), c.argmax():n-c[::-1].argmax(), 0:3]
    else:
        out = np.empty((0,0),dtype=bool)
    return out

def resize(image):
    return cv2.resize(image, dsize=(100,100), interpolation = cv2.INTER_CUBIC)
    

def main(argv):
    ###Lets load in our df of files and bits ###
    DF = pd.read_pickle("MProjDF.pk1")
    # print(list(DF.columns))
    # print(DF)

    for iter in range(len(DF)):
    # for iter in range(3):
        print("Working on: ", DF.index[iter])
        
        print("loading image")
        img = np.asarray(read_raw_img(DF.image_path[iter]))
        # print("image loaded, dimesnions: ", img.shape)

        print("loading mask")
        mask = np.load(DF.mask_path[iter])
        # print("mask loaded, dimensions: ", mask.shape)

        ### Making RGB channeled mask ###
        mask_RBG = np.dstack([mask,mask,mask])

        ### Multiplyin mask with image ###
        print("Masking image")
        masked_img = np.multiply(img,mask_RBG)

        ### trim the spare zeros with a square crop ###
        print("trimming the edges")
        cropped = smallestbox(a = mask, img = masked_img)
        #print(cropped)

        print("resizing image")
        resized_img = np.array(resize(cropped))
        img_normalized = resized_img / 255
        
        # ### Manual normalisation of smaller images ###
        # print("normalising image")
        # for i in range(3):
        #     # print("Channel: ", i)
        #     img_normalized[i] = resized_img[i] - np.mean(resized_img[i])
        #     img_normalized[i] = img_normalized[i] / np.std(resized_img[i])
        #     # print("Normalised mean: ", np.mean(img_normalized[i]))
        #     # print("Stdev:", np.std(img_normalized[i]))

        print("saving as .npy")
        save_str =  "sq_non_norm/" + str(DF.index[iter])[2:-3] + "_sq.npy"
        print("savestring: ", save_str)
        np.save(save_str, img_normalized)


    return
    
# This function makes sure the boilerplate in full when called from the terminal, then passes control to main function 
if __name__ == "__main__":
    """ Makes sure the "main" function is called from command line """
    status = main(sys.argv)
    sys.exit(status)