#!/usr/bin/env python3

"""Used to generate masks on images to extract flower from background"""

__appname__ = 'dataframe.py'
__author__ = '[Elliott Parnell (ejp122@ic.ac.uk)]'
__version__ = '0.0.1'

import sys
import os
import glob
import numpy as np
import pandas as pd
import re
import pickle
import csv

def main(argv):
    """ Main entry point of the program """
    #Get list of images
    imagelist = glob.glob("../images/renamedimages/*.CR2")
    print("Number of images in folder: ",len(imagelist))
    
    # Get list of the masks
    masklist = glob.glob("npy_masks/*.npy")
    print("Number of masks in folder: ", len(masklist))

    #Initiate empty image name list
    imagename_noEXT = []
    
    # Strip image path to just file name in order to match with masks
    for iter in range(len(imagelist)):
        tempTuple1 = os.path.split(imagelist[iter])
        image_stem = tempTuple1[0]
        imagename = tempTuple1[1]
        tempTuple = os.path.splitext(imagename)
        imagename_noEXT.append(tempTuple[0][0:15])
    
    
    #Initiate empty masks name list
    maskname_noEXT = []

    #Strip mask name down for matching
    for iter in range(len(masklist)):
        tempTuple1 = os.path.split(masklist[iter])
        path_mask = tempTuple1[0]
        maskname = tempTuple1[1]
        tempTuple = os.path.splitext(maskname)
        if (tempTuple[0][14] == "_"):
            maskname_noEXT.append(tempTuple[0][0:14])
        else:    
            maskname_noEXT.append(tempTuple[0][0:15])


    #print(maskname_noEXT)

    matches = list(set(imagename_noEXT) & set(maskname_noEXT))

    
    print("Number of matches: ",len(matches))
    image_path = [None]*len(maskname_noEXT)
    mask_score = [None]*len(maskname_noEXT)
    c_x = [None]*len(maskname_noEXT)
    plant_id = [None]*len(maskname_noEXT)
    flower_id = [None]*len(maskname_noEXT)
    unique_id = [None]*len(maskname_noEXT)
    pol_stat = [None]*len(maskname_noEXT)
    day = [None]*len(maskname_noEXT)
    t_step = [None]*len(maskname_noEXT)
    pol_res = [None]*len(maskname_noEXT)
    experiment_clock = [None]*len(maskname_noEXT)
    pol_outcome = [None]*len(maskname_noEXT)

    

    with open("pol_outcome.csv", mode="r") as csvfile:
        reader = csv.reader(csvfile)
        pol_dict = {rows[0]:rows[1] for rows in reader}


    for iter in range(len(maskname_noEXT)):
        #Create the path to Image for each mask
        image_path[iter] = image_stem + "/" + maskname_noEXT[iter] +".CR2"

        #Extract mask score 
        mask_score[iter] = str(re.findall(r"(_[0-9]+\.)", masklist[iter])[0])
        mask_score[iter] = mask_score[iter][1:(len(mask_score[iter])-1)]
        expression = "^[_](\d+)[.]$"
        if mask_score[iter].startswith(("0","1")):
            mask_score[iter] = float( "1."+ mask_score[iter])
        elif mask_score[iter].startswith(("9","8","7","6","5","4","3","2")):
            mask_score[iter] = float( "0."+ mask_score[iter])

        # Control or expirmental 
        if (maskname_noEXT[iter][0]=="X"):
            c_x[iter]= "Experimental"
        elif (maskname_noEXT[iter][0]=="C"):
            c_x[iter]= "Control"

        plant_id[iter]=int(maskname_noEXT[iter][2:4])
        flower_id[iter]=int(maskname_noEXT[iter][5:7])
        unique_id[iter]=(maskname_noEXT[iter][2:7])
        pol_stat[iter]=maskname_noEXT[iter][8]
        day[iter]=int(maskname_noEXT[iter][10:12])
        t_step[iter]=int(maskname_noEXT[iter][13])
        experiment_clock[iter]= ((day[iter]-1)*24)+((t_step[iter]-1)*2)
        pol_outcome[iter]= pol_dict[unique_id[iter]]
        
    
    
    dflist = zip(masklist,image_path, mask_score, c_x, plant_id, flower_id, unique_id, pol_stat, day, t_step, experiment_clock, pol_outcome)
    colnames = ["mask_path","image_path", "mask_score", "C_X", "plant_id", "flower_id", "unique_id", "pol_stat", "day", "t_step", "exp_clock", "pol_outcome"]

    print("Creating Dataframe")
    df = pd.DataFrame(dflist, index =[maskname_noEXT], columns =colnames)
    pd.set_option('display.max_columns', None)
    print(df.head())

    df.to_pickle("MProjDF.pk1")

    return

# This function makes sure the boilerplate in full when called from the terminal, then passes control to main function 
if __name__ == "__main__":
    """ Makes sure the "main" function is called from command line """
    status = main(sys.argv)
    sys.exit(status)