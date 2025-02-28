import numpy as np
import cv2 as cv
import os
from stiliRzare import apply_hsv_filter
from crop import find_corners as find_corners
from diferenta import diference

src = r"D:\surse\Python\cava\CAVA-2024-Tema1\antrenare"
dst = r"D:\surse\Python\cava\CAVA-2024-Tema1\rezultate"
#Empty table so we can find the first piece
img_init = cv.imread(r"D:\surse\Python\cava\CAVA-2024-Tema1\imagini_auxiliare\01.jpg")

#The HSV values for the filter used to find the corner pieces of the table
lower_bound= np.array([21,150,199])
upper_bound= np.array([30,255,255])
#The HSV values for the filter used to highlight the pieces
lower_bound_pieces = np.array([0,0,165])
upper_bound_pieces = np.array([82,94,255])

#All the operations so that the empty table is ready to be compared with the other images
img1_init_filter=apply_hsv_filter(img_init, lower_bound, upper_bound)

top_left, bottom_right = find_corners(img1_init_filter)
img_init = img_init[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
img_init = cv.resize(img_init, (1470, 1470), interpolation = cv.INTER_AREA)

img_init= cv.cvtColor(img_init, cv.COLOR_BGR2HSV)
mask = cv.inRange(img_init, lower_bound_pieces, upper_bound_pieces)
img_init = cv.bitwise_and(img_init, img_init, mask=mask)

img1=img_init
i=0

# iterate over the images in the source folder
for filename in os.listdir(src):
    if filename.endswith(".jpg"):

        i+=1
        img2 = cv.imread(os.path.join(src, filename))

        img_filter = apply_hsv_filter(img2, lower_bound, upper_bound)

        top_left, bottom_right = find_corners(img_filter)
        img_crop = img2[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        img2_crop = cv.resize(img_crop, (1470, 1470), interpolation = cv.INTER_AREA)
        
        img2 = cv.cvtColor(img2_crop, cv.COLOR_BGR2HSV)
        mask = cv.inRange(img2, lower_bound_pieces, upper_bound_pieces)
        img2 = cv.bitwise_and(img2, img2, mask=mask)
        
        coord=diference(img1, img2)
        coord_show = (coord[0]+1,chr(coord[1]+ord('A')))

        #Create a .txt file which containts the position of the piece 
        with open(os.path.join(dst, filename[:-4]+".txt"), "w") as f:
            f.write(str(coord_show[0])+coord_show[1])                        
        print(i)
        print(coord_show)
                
    #If we have 50 images we need to reinitialize the first image
    if i==50:
        img1 = img_init
        i=0
        img2 = img_init
    else:
        img1= img2
