#!/bin/bash

#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=1:mem=4gb

# This is my shell script to run the python script to generate masks
# Each iteration of this script will run the script python for a single image 
# That python script will generate and save the arrays - can delete rubish ones later 

# Load enivironment 




# Copy my files to my temporary directory 
echo "Moving to temp directory"
mv $HOME/images $
echo "creating temp output folder
mkdir outputs

# Run python script 
echo "Running python"
python3 $HOME/code/masks_HPC.py
echo "Python finished running"

# Copy required files back 
pwd
echo "Moving output files"
mv *.npy $HOME/output_files