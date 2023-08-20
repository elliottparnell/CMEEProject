#!/usr/bin/env python3

"""Used to generate masks on images to extract flower from background"""

__appname__ = 'masks_HPC.py'
__author__ = '[Elliott Parnell (ejp122@ic.ac.uk)]'
__version__ = '0.0.1'

import sys
import masks_main
from masks_main import generate_masks
import os
import glob

def main(argv):
    """ Main entry point of the program """
    sys.path.insert(0, "/rds/general/user/ejp122/home/code")

    iter = int(os.getenv("PBS_ARRAY_INDEX"))
    print("PBS ARRAY INDEX: ", iter)
    # filelist = glob.glob("/rds/general/user/ejp122/home/images/*.CR2")
    TEMPDIR = os.environ.get("$TMPDIR")
    filelist = glob.glob(TEMPDIR)
    print("files to have masks generated: ", len(filelist))
    # for iter in range(len(filelist)):
        # generate_masks(filelist[iter])
    generate_masks(filelist[iter])
    return

# This function makes sure the boilerplate in full when called from the terminal, then passes control to main function 
if __name__ == "__main__":
    """ Makes sure the "main" function is called from command line """
    status = main(sys.argv)
    sys.exit(status)