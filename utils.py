from PIL import Image
import os
import random
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
import scipy.misc as misc
def load_images(image_path):
    img = plt.imread(image_path)
    return np.expand_dims(img,2)

def create_batches(path):

    x = []
    y = []
    for i in range(len(path)):

        x_image=load_images(path[i][0])
        y_image=load_images(path[i][1])
        prob = np.random.uniform()
        if prob > 0.5:
            x_image= np.fliplr(x_image)
            y_image=np.fliplr(y_image)
        x.append(x_image)
        y.append(y_image)
    return x,y

def test_batch(batch_size,mode='test/'):
    image_list = os.listdir('test2/original')
    batches=[]
    x = []
    y = []
    for i in range(batch_size):

        x_image=load_images(mode+'original/'+'%d.png'%i)
        # y_image=load_images(mode+'target/'+image_list[i])


        x.append(x_image)
        # y.append(y_image)
    return x
# def image_resize():
#     if os.path.exists('korean/original') == False:
#         os.mkdir('korean/original')
#     if os.path.exists('korean/target') == False:
#         os.mkdir('korean/target')
#     size = 256 , 256
#     list=os.listdir('A')
#     list2 = os.listdir('B')
#     list3 = [img for img in list2 if img in list and img.startswith('uni')]
#
#     if len(list) >= len(list3):
#         list=list3
#
#     for file in list:
#         target_image = cv2.imread('A/'+file,cv2.IMREAD_GRAYSCALE)
#         resized_target_image = cv2.resize(target_image,(256,256))
#         cv2.imwrite('korean/original/'+file,resized_target_image)
#
#     for file in list:
#         target_image = cv2.imread('B/'+file,cv2.IMREAD_GRAYSCALE)
#         resized_target_image = cv2.resize(target_image,(256,256))
#         cv2.imwrite('korean/target/'+file,resized_target_image)
#
#
# def move_train():
#
#     if os.path.exists('test3/original') == False:
#         os.mkdir('test3/original')
#     if os.path.exists('test3/target') == False:
#         os.mkdir('test3/target')
#
#     img=os.listdir('korean/target')
#     random.shuffle(img)
#     for i in range(len(img)-2000):
#
#         shutil.move('korean/original/'+img[i],'test3/original/'+img[i])
#         shutil.move('korean/target/'+img[i],'test3/target/'+img[i])