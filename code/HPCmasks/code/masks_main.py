#!/usr/bin/env python3

""" Mask generation helper functions"""

### Imports required for functions ###
import torch
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import matplotlib.pyplot as plt
import rawpy
import cv2

### Load raw image function ###
def read_raw_img(filepath):
    import rawpy
    """Reads a raw image format and outputs post proccessing image"""
    try:
        with rawpy.imread(filepath) as raw:
            processed_img = raw.postprocess()
        return processed_img
    except:
        print("Please provide raw image format e.g. CR2")
        return 
    
### Show mask on image ###
def show_mask(mask, ax, random_color=False):
    import numpy as np
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

### Show points on image ###    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

### Show box on image ###    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

# ### This function loads the model ###
# def load_SAM():
#     from segment_anything import SamPredictor, sam_model_registry
#     DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     MODEL_TYPE = "vit_h"
#     CHECKPOINT_PATH = "sam_vit_h_4b8939.pth" # This path may need to be edited

#     print("Registering Model \n")
#     sam = ssh o(device=DEVICE)

#     print("Loading Mask predictor")
#     return SamPredictor(sam)

# ### This function loads in the image and sets image with SAM ###
# def load_image(filepath):
#     import cv2
#     print("Loading image \n")
#     image_bgr = read_raw_img(filepath)
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

#     print("Setting image")
#     return mask_predictor.set_image(image_rgb)
    

### THIS IS MY MAIN FUNCTION TO BE CALLED BY THE CLUSTER.py script ###
def generate_masks(filepath):

    import torch
    from segment_anything import SamPredictor, sam_model_registry
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import os

    ### Section for preparing the items needed for filenaming
    tempTuple = os.path.split(filepath)
    path_name = tempTuple[0]
    filename = tempTuple[1]
    tempTuple = os.path.splitext(filename)
    filename_noEXT = tempTuple[0]
    

    # Load the model 
    #mask_predictor = load_SAM()
    from segment_anything import SamPredictor, sam_model_registry
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = "/rds/general/user/ejp122/home/code/sam_vit_h_4b8939.pth" # This path may need to be edited

    print("Registering Model \n")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)

    print("Loading Mask predictor")
    mask_predictor = SamPredictor(sam)

    # Load the image 
    # load_image(filepath)
    print("Loading image \n")
    image_bgr = read_raw_img(filepath)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("Setting image")
    mask_predictor.set_image(image_rgb)
    print("Image set")

    ### Define the input boxes ###
    # Acess using referencing # input_boxes[0]
    # Format (x1,y1,x2,y2)
    input_boxes = np.array([[850,500,2500,3500],
                            [200,200,3000,3700],
                            [200,200,3500,3700],
                            [200,200,4300,3700],
                            [200,200,5300,3700],
                            [1000,1000,3500,3000],
                            [800,800,4000,3000],
                            [1500,1000,4000,3000],
                            [1000,1500,3700,3250]])
    
    # mask_path = "../outputs/"
    # TEMPDIR = os.environ.get("$TMPDIR")
    # mask_path = TEMPDIR + "outputs/"
    # Storage for masks and scores
    mask_store = []
    score_store = []
    logit_store = []
    output_path_store = []
        
    for iter in range(len(input_boxes)):
        
        # Mask store: 0 = mask, 1 = score, 2 = logita
        temp_mask, temp_score, temp_logit = mask_predictor.predict(
        box=input_boxes[iter],
        multimask_output=False
        )
        mask_store.append(temp_mask[0])
        score_store.append(temp_score[0])
        logit_store.append(temp_logit[0])
        
        maskname = '_'.join([filename_noEXT, "mask",str(iter),str(round(score_store[iter],3))[2:]])        
        # output_path =  mask_path + maskname
        output_path_store.append(maskname)

        print("Masks ", iter, " generated")
        #print("score: ", score_store[iter])

    print("score_store: ", score_store)

    # from matplotlib import pyplot as plt
    # for item in range(len(score_store)):
    #     plt.imshow(mask_store[item])
    #     print("Score: ", score_store[item])
    #     plt.show()

    best_mask_index = score_store.index(max(score_store))
    # best_mask = mask_store[best_mask_index]
    np.save(output_path_store[best_mask_index],mask_store[best_mask_index])

    
    
