#!/usr/bin/env python3

""" Reads in images, selects box around mask and resized, feedsthat image into a fourier transform for RGB and saves each into DF"""

__appname__ = 'image_2_powerspec.py'
__author__ = '[Elliott Parnell (ejp122@ic.ac.uk)]'
__version__ = '0.0.1'


### Imports ###
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
import scipy.stats as stats


### FUNCTIONS ###

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
    return cv2.resize(image, dsize=(1000,1000), interpolation = cv2.INTER_CUBIC)

def gen_power_spec(image):
    npixH=image.shape[0]
    npixV=image.shape[1]
    if npixH == npixV:
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image)**2

        kfreq = np.fft.fftfreq(npixH) * npixH
        kfreq2D = np.meshgrid(kfreq, kfreq)
        knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

        knrm = knrm.flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()

        kbins = np.arange(0.5, npixH//2+1, 1.)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                            statistic = "mean",
                                            bins = kbins)
        Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

        return(Abins)
    else:
        print("ERROR: Image not square")
        return(0)
    
def seq_gen(string,len):
    return_list = []
    for iter in range(len):
        item = string + str(iter+1)
        return_list.append(item)
    return return_list


    

#### MAIN FUNCTION ####
def main(argv):
    DF = pd.read_pickle("MProjDF.pk1")

    Rseq = seq_gen("R", 500)
    Bseq = seq_gen("B", 500)
    Gseq = seq_gen("G", 500)
    colnames = Rseq + Bseq + Gseq

    saveDF = pd.DataFrame(columns=colnames)

    for iter in range(len(DF)):
    # for iter in range(3):
        print("Working on: ", DF.index[iter])
        
        print("loading image")
        img = np.asarray(read_raw_img(DF.image_path[iter]))

        print("loading mask")
        mask = np.load(DF.mask_path[iter])
        # print("mask loaded, dimensions: ", mask.shape)

        ### Making RGB channeled mask ###
        mask_RGB = np.dstack([mask,mask,mask])

        ### Multiplyin mask with image ###
        print("Masking image")
        masked_img = np.multiply(img,mask_RGB)

        ### trim the spare zeros with a square crop ###
        print("trimming the edges")
        cropped = smallestbox(a = mask, img = masked_img)
        #print(cropped)

        print("resizing image")
        resized_img = np.array(resize(cropped))
        img_normalized = resized_img / 255

        ###### DO I NEED TO NORAMLISE ### ?

        image = np.moveaxis(resized_img, -1, 0)
        
        DF_row = []

        DF_row.extend(gen_power_spec(image[0]))
        DF_row.extend(gen_power_spec(image[1]))
        DF_row.extend(gen_power_spec(image[2]))
        

        DF_index = str(DF.index[iter])[2:-3]

        saveDF.loc[DF_index] = DF_row

    #print(saveDF)
    saveDF.to_pickle("powerspec.pk1")








    return

# This function makes sure the boilerplate in full when called from the terminal, then passes control to main function 
if __name__ == "__main__":
    """ Makes sure the "main" function is called from command line """
    status = main(sys.argv)
    sys.exit(status)