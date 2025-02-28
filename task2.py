import numpy as np
import cv2 as cv
import os
from stilizare import apply_hsv_filter
from crop import find_corners as find_corners
from diferenta import diference

src = r"D:\surse\Python\cava\CAVA-2024-Tema1\antrenare"
dst = r"D:\surse\Python\cava\CAVA-2024-Tema1\rezultate"
template = r"D:\surse\Python\cava\CAVA-2024-Tema1\img_comp\result\results"
#Empty table so we can find the first piece
img_init = cv.imread(r"D:\surse\Python\cava\CAVA-2024-Tema1\imagini_auxiliare\01.jpg")

#The HSV values for the filter used to find the corner pieces of the table
lower_bound= np.array([21,150,199])
upper_bound= np.array([30,255,255])
#The HSV values for the filter used to select the number on the piece
lower_bound_piece = np.array([0, 0, 0])
upper_bound_piece = np.array([190, 255, 70])
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


        #We take the square which changed
        square = img2_crop[coord[0] * 105:(coord[0] + 1) * 105, coord[1] * 105:(coord[1] + 1) * 105]
        
        #Apply the filter
        mask = cv.inRange(square, lower_bound_piece, upper_bound_piece)
        square = cv.bitwise_and(square, square, mask=mask)
      
        #mask = cv.GaussianBlur(square, (5, 5), 0)
        #_, mask = cv.threshold(mask, 100, 255, cv.THRESH_BINARY)
       

        # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # mask = cv.filter2D(mask, -1, kernel)
       

        kernel = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)


        #mask = cv.Canny(mask, 100, 200)

        #Remove 10 pixels from the edges
        mask = mask[12:93, 12:93]

        _, mask = cv.threshold(mask, 100, 255, cv.THRESH_BINARY)

        # cv.imshow("square", mask)
        # cv.waitKey(0)
        # cv.destroyAllWindows()



        y_nonzero, x_nonzero = np.nonzero(mask)
        
        top = np.min(y_nonzero)
        bottom = np.max(y_nonzero)
        left = np.min(x_nonzero)
        right = np.max(x_nonzero)

        # Crop the image to these boundaries
        mask = mask[top:bottom+1, left:right+1]

        # cv.imshow("square", mask)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        #Resize
        mask = cv.resize(mask, (105, 105), interpolation = cv.INTER_AREA)

        square = mask
        

        #Save the square
        #cv.imwrite(os.path.join(dst, filename), square)

        #Maximum corelation and maximum corelation position
        cor_max = -1
        pos_max = -1

    
        # Compare the square with every photo from the template folder
        for filename2 in os.listdir(template):
            if filename2.endswith(".jpg"):
                # Load the image
                img = cv.imread(os.path.join(template, filename2))
                
                # Convert the image to grayscale
                img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                # Apply the template matching
                result = cv.matchTemplate(square, img_gray, cv.TM_CCOEFF_NORMED)
                if result.max() > cor_max:
                    cor_max = result.max()
                    pos_max = filename2[:-4]
                
                
        # Create a .txt file which contains the position of the piece and the name of the image most similar to
        with open(os.path.join(dst, filename[:-4]+".txt"), "w") as f:
            f.write(str(coord_show[0]) + coord_show[1])
            f.write(f" {pos_max}")
        # print(coord_show)
        # print(pos_max)    

    #If we have 50 images we need to reinitialize the first image
    if i==50:
        img1 = img_init
        i=0
        img2 = img_init
    else:
        img1= img2
    # if i==15:
    #     break
