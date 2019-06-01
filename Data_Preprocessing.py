
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys


# In[2]:


f = open('Practice/Datasets/wider_face_split/wider_face_train_bbx_gt.txt', 'r')
lines = f.readlines()
print(np.shape(lines))


# In[3]:


lines[:4]
image_path = "C:/Users/sport/Desktop/Practice/Datasets/WIDER_train/WIDER_train/images/"
print(os.path.join(image_path, lines[0]))
img = cv2.imread(os.path.join(image_path, lines[0][:-1]))
print(np.shape(img))


# In[4]:


image_list = []
num_images = []
for i in range(len(lines)):
    if "jpg" in lines[i]:
        image_list.append(lines[i][:-1])
        num_images.append(lines[i+1][:-1])
print(np.shape(image_list), np.shape(num_images))


# In[6]:


line_count = 0
save_path_face = 'C:/Users/sport/Desktop/CV_Proj2/training_face/'

for i in range(len(image_list)):
    img = cv2.imread(os.path.join(image_path, image_list[i]))
    line_count = line_count+2
    for j in range(int(num_images[i])):
        if(int(num_images[i])<4):
            f = lines[line_count]
            x,y,w,h = map(int,f.split(" ")[0:4])
            if(w>10 and h>10):
                cv2.imwrite(os.path.join(save_path_face,'img{}.jpg'.format(line_count-2*(i+1))),cv2.resize(img[y:y+h,x:x+w],(60,60)))
        line_count = line_count + 1        


# In[17]:


line_count = 0
save_path_nonface = 'C:/Users/sport/Desktop/CV_Proj2/training_nonface/'

for i in range(len(image_list)):
    img = cv2.imread(os.path.join(image_path, image_list[i]))
    line_count = line_count+2
    f = lines[line_count]
    x,y,w,h = map(int,f.split(" ")[0:4])
    if(x>5 and y>5):
        cv2.imwrite(os.path.join(save_path_nonface,'imgyx1{}.jpg'.format(line_count-2*(i+1))),cv2.resize(img[0:y,0:x],(60,60)))
    line_count = line_count + int(num_images[i])        

