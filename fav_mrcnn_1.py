#!/usr/bin/env python
# coding: utf-8

# In[3]:


import fav


# In[4]:



import os
import sys
import json
import datetime
import numpy as np
import skimage
import skimage.draw
import matplotlib.pyplot as plt

from glob import glob
import random


# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


# In[ ]:





# In[5]:


xy = fav.favConfig()


# In[6]:


xyd =fav.favDataset()


# In[7]:


xyd.load_fav('te-pam','train', 'via_export_json.json' )


# In[8]:


xyd.prepare()


# In[9]:


zz = xyd.load_mask(1)


# In[ ]:





# In[10]:


zz[1]


# In[11]:


plt.figure(figsize= (20,20))
plt.imshow(zz[0][:,:,0])


# In[12]:


from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# In[13]:


model = modellib.MaskRCNN(mode='training', config=xy, model_dir=ROOT_DIR)
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])


# In[14]:


LEARNING_RATE = 0.006


# In[15]:


PRETRAINED_MODEL_PATH = "./fav20200325T1329/mask_rcnn_fav_0002.h5"
model = modellib.MaskRCNN(mode='training', config=xy, model_dir='./fav20200325T1329')
model_path = PRETRAINED_MODEL_PATH
model.load_weights(model_path, by_name=True)


# In[ ]:


model.train(xyd, xyd,
            learning_rate=LEARNING_RATE,
            epochs=6,
            layers='all',
            augmentation=None)
 
new_history = model.keras_model.history.history


# In[ ]:


class InferenceConfig(fav.favConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# In[ ]:


PRETRAINED_MODEL_PATH = "./fav20200325T0732/mask_rcnn_fav_0002.h5"
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./fav20200325T0732')
model_path = PRETRAINED_MODEL_PATH
model.load_weights(model_path, by_name=True)


# In[ ]:


file_names = glob(os.path.join('te-pam/train/', "*.png"))
class_names = ['BG', 'bond', 'Wire']
test_image = skimage.io.imread(file_names[random.randint(0,len(file_names)-1)])
predictions = model.detect([test_image], verbose=1) # We are replicating the same image to fill up the batch_size
p = predictions[0]
visualize.display_instances(test_image, p['rois'], p['masks'], p['class_ids'], 
                            class_names, p['scores'])


# In[ ]:




