import numpy as np
import matplotlib.pyplot as plt
import cv2

img_file = cv2.imread('F:\CVIP Projects\Project1/histogram.jpg', 0)
cv2.imshow('Original Image',img_file)
height, width = img_file.shape

#From the Histogram Equalization Algorithm
#Step 1:

H_img = np.zeros(256, dtype = np.int)
G = np.zeros(256, dtype = np.int)
H_cum = np.zeros(256, dtype = np.int)
H_output_img = np.zeros(256, dtype = np.int)
transform = np.zeros(256, dtype = np.int)
final_img = np.zeros((height, width), np.uint8)

#Step 2:

def histogram():
    for i in range(0,256):
        H_img[i] = i
    
    for i in range(0, height):
        for j in range(0, width):
            gp = img_file[i][j]
            G[gp] = G[gp] + 1
              
    plt.figure('Original Image Histogram')
    plt.xlabel('gray-levels')
    plt.ylabel('number of pixels')
    plt.title('Original_Image_Histogram')
    plt.plot(H_img, G)
    plt.show()
    return

#Step 3:

def cumulative():
    H_cum[0] = G[0]
    
    for i in range(0,256):
        H_img[i] = i
    
    for i in range(1, 256):
        H_cum[i] = H_cum[i-1] + G[i]
    
    plt.figure('Cumulative Histogram')
    plt.xlabel('Gray-levels')
    plt.ylabel('Total pixels')
    plt.title('Cumulative_Histogram')
    plt.plot(H_img, H_cum)
    plt.show()
    return
    
#Step 4:

def Transformation():
    for i in range(0,256):
        H_img[i] = i
    
    for i in range(0,256):
        transform[i] = np.round((255*H_cum[i])/img_file.size)
    
    plt.figure('Transformation')
    plt.xlabel('Gray-levels')
    plt.ylabel('transform')
    plt.title('Transformation_function')
    plt.plot(H_img, transform)
    plt.show()
    
    return transform

#Step 5:

def output_histogram(transform):
    for i in range(0,256):
        H_img[i] = i
    
    for i in range(0, height):
        for j in range(0, width):
            gp = img_file[i][j]
            final_img[i][j] = transform[gp]
    
    for i in range(0, height):
        for j in range(0, width):
              gp = final_img[i][j]
              H_output_img[gp] = H_output_img[gp] + 1
    
    plt.figure('Output Image Histogram')
    plt.xlabel('Gray-levels')
    plt.ylabel('Number of pixels')
    plt.title('Output_Image_Histogram')
    plt.plot(H_img, H_output_img)
    plt.show()
    return final_img
    
histogram()
cumulative()
transform_function = Transformation()
final_image = output_histogram(transform_function)

cv2.imshow('Final Image', final_image)
  
