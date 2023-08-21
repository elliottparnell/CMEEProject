# CMEEProject

Welcome to my git containing all the code used for my CMEE project on detecting floral senescene and pollination using machine learning and image data.

If you have any questions or would like access to the photo dataset please email: elliott.parnell22@imperial.ac.uk

## Project workflow 
### Step 1: Rename images
Use barcode-QRcodeScannerPy_vRAW.py to rename images based on qr code in image

### Step 2: Create image masks
Use code in HPCmasks folder to create background masks for images. This is designed to be run on a HPC. This will need SegmentAnythingModel checkpoint file available from: https://github.com/facebookresearch/segment-anything

Masks created should be saved in npy_masks directory

### Step 3: Create dataframe 
dataframe.py reads the image and mask files and creates a data frame of matching images, masks and the data contained in each QR code. This will save in code directory

### Step 4: Image preprocessing
Use square_npy_non_norm_generator.py to crop and resize images based of their masks

### Step 5: Generate power spectra
From images generate power spectra, these will be saved in a dataframe in the code directory

### Step 6: Machine learning
Run Data_exp_ML.R Machine learning using the power spectra data, and CNN_flower_age.py and CNN_treatment.py for CNNs using the image data. 