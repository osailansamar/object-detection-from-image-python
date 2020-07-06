#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import required libraries
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox 


# In[4]:


# image to be sccaned
img = cv2.imread('exmp.jpg')

#detect objects
bbox, label, conf = cv.detect_common_objects(img)
#surround objects with a rectangle and label
rslt = draw_bbox(img, bbox, label, conf)

#show result
plt.imshow(rslt)
plt.show()


# In[ ]:




