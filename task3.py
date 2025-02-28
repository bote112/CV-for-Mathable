import numpy as np
import cv2 as cv
import os
from stilizare import apply_hsv_filter
from crop import find_corners as find_corners
from diferenta import diference

src = r"D:\surse\Python\cava\CAVA-2024-Tema1\antrenare"
dst = r"D:\surse\Python\cava\CAVA-2024-Tema1\rezultate"
template = r"D:\surse\Python\cava\CAVA-2024-Tema1\img_comp"

#Empty table so we can find the first piece
img_init = cv.imread(r"D:\surse\Python\cava\CAVA-2024-Tema1\dsadasdimagini_auxiliare\01.jpg")

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

#Initial matrix is all -1
matrix_init = np.full((14, 14), -1)
matrix_init[6][6]=1
matrix_init[6][7]=2
matrix_init[7][6]=3
matrix_init[7][7]=4

matrix_init[0][0]=-3
matrix_init[0][13]=-3
matrix_init[13][0]=-3
matrix_init[13][13]=-3
matrix_init[0][6]=-3
matrix_init[0][7]=-3
matrix_init[6][0]=-3
matrix_init[7][0]=-3
matrix_init[13][6]=-3
matrix_init[13][7]=-3
matrix_init[6][13]=-3
matrix_init[7][13]=-3

matrix_init[1][1]=-2
matrix_init[1][12]=-2
matrix_init[12][1]=-2
matrix_init[12][12]=-2
matrix_init[2][2]=-2
matrix_init[2][11]=-2
matrix_init[11][2]=-2
matrix_init[11][11]=-2
matrix_init[3][3]=-2
matrix_init[3][10]=-2
matrix_init[10][3]=-2
matrix_init[10][10]=-2
matrix_init[4][4]=-2
matrix_init[4][9]=-2
matrix_init[9][4]=-2
matrix_init[9][9]=-2

img1=img_init
contor=int(1)
matrix=matrix_init
game=1
player="Player1"
player_limit=1
turns_list = []
sum=0
#we open with w all 4 files so they are empty
with open(os.path.join(dst, "1_scores.txt"), "w") as scores_file:
    pass
with open(os.path.join(dst, "2_scores.txt"), "w") as scores_file:
    pass
with open(os.path.join(dst, "3_scores.txt"), "w") as scores_file:
    pass
with open(os.path.join(dst, "4_scores.txt"), "w") as scores_file:
    pass

# iterate over the images in the source folder
for filename in os.listdir(src):
    if contor==1:
        turns_list=[]
        player="Player1"
        player_limit=1
        #We search for the txt file named {game}_turns.txt
        turns_file_path = os.path.join(src, f"{game}_turns.txt")
        if os.path.exists(turns_file_path):
            with open(turns_file_path, "r") as f:

                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        player = parts[0]
                        turn = int(parts[1])
                        turns_list.append([player, turn])
    if filename.endswith(".jpg"):

        if len(turns_list)!=1:
            if (contor>=turns_list[1][1]):
                
                #We write in the file the current player, the turn and the sum of the points
                with open(os.path.join(dst, f"{game}_scores.txt"), "a") as scores_file:
                    scores_file.write(f"{turns_list[0][0]} {turns_list[0][1]} {sum}\n")
                sum=0
                turns_list.pop(0)
                player=turns_list[0][0]
                player_limit=turns_list[0][1]
        
        
    

        contor+=1
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
    
      

        kernel = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)



        #Remove 10 pixels from the edges
        mask = mask[12:93, 12:93]

        _, mask = cv.threshold(mask, 100, 255, cv.THRESH_BINARY)


        y_nonzero, x_nonzero = np.nonzero(mask)
        
        top = np.min(y_nonzero)
        bottom = np.max(y_nonzero)
        left = np.min(x_nonzero)
        right = np.max(x_nonzero)

        # Crop the image to these boundaries
        mask = mask[top:bottom+1, left:right+1]

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

            
    
        x,y=coord
        value=int(pos_max)

        multiplier=1
        #We check if the value on the position is -2 or -3
        if matrix[x][y]==-2:
            multiplier=2
        elif matrix[x][y]==-3:
            multiplier=3
        
        #We check every direction for 2 steps if there are pieces with positive values
        #If we find 2 pieces that add,absdiff,multiply or divide and equal to value
        #We add to sum value*multiplier

        #If the value is 0 we don't need to check because we wont add any points
        if value!=0:
            #Up
            if x-2>=0:
                if matrix[x-1][y]!=-1 and matrix[x-2][y]!=-1 and matrix[x-1][y]!=-2 and matrix[x-2][y]!=-2 and matrix[x-1][y]!=-3 and matrix[x-2][y]!=-3:
                    if (matrix[x-1][y]+matrix[x-2][y]==value or 
                    abs(matrix[x-1][y]-matrix[x-2][y])==value or 
                    matrix[x-1][y]*matrix[x-2][y]==value or 
                    (matrix[x-1][y]!=0 and matrix[x-2][y]!=0 and (matrix[x-1][y]/matrix[x-2][y]==value or matrix[x-2][y]/matrix[x-1][y]==value))):
                        sum+=value*multiplier
            #Down
            if x+2<=13:
                if matrix[x+1][y]!=-1 and matrix[x+2][y]!=-1 and matrix[x+1][y]!=-2 and matrix[x+2][y]!=-2 and matrix[x+1][y]!=-3 and matrix[x+2][y]!=-3:
                    if (matrix[x+1][y]+matrix[x+2][y]==value or 
                    abs(matrix[x+1][y]-matrix[x+2][y])==value or 
                    matrix[x+1][y]*matrix[x+2][y]==value or 
                    (matrix[x+1][y]!=0 and matrix[x+2][y]!=0 and (matrix[x+1][y]/matrix[x+2][y]==value or matrix[x+2][y]/matrix[x+1][y]==value))):
                        sum+=value*multiplier

            #Left
            if y-2>=0:
                if matrix[x][y-1]!=-1 and matrix[x][y-2]!=-1 and matrix[x][y-1]!=-2 and matrix[x][y-2]!=-2 and matrix[x][y-1]!=-3 and matrix[x][y-2]!=-3:
                    if (matrix[x][y-1]+matrix[x][y-2]==value or 
                    abs(matrix[x][y-1]-matrix[x][y-2])==value or 
                    matrix[x][y-1]*matrix[x][y-2]==value or 
                    (matrix[x][y-1]!=0 and matrix[x][y-2]!=0 and (matrix[x][y-1]/matrix[x][y-2]==value or matrix[x][y-2]/matrix[x][y-1]==value))):
                        sum+=value*multiplier

            #Right
            if y+2<=13:
                if matrix[x][y+1]!=-1 and matrix[x][y+2]!=-1 and matrix[x][y+1]!=-2 and matrix[x][y+2]!=-2 and matrix[x][y+1]!=-3 and matrix[x][y+2]!=-3:
                    if (matrix[x][y+1]+matrix[x][y+2]==value or 
                    abs(matrix[x][y+1]-matrix[x][y+2])==value or 
                    matrix[x][y+1]*matrix[x][y+2]==value or 
                    (matrix[x][y+1]!=0 and matrix[x][y+2]!=0 and (matrix[x][y+1]/matrix[x][y+2]==value or matrix[x][y+2]/matrix[x][y+1]==value))):
                        sum+=value*multiplier
        matrix[x][y]=value
        
        #If we have 50 images we need to reinitialize the first image
        if contor>50:
            img1 = img_init
            with open(os.path.join(dst, f"{game}_scores.txt"), "a") as scores_file:
                scores_file.write(f"{turns_list[0][0]} {turns_list[0][1]} {sum}\n")
            contor=1
            game+=1
            img2 = img_init
            
            matrix=matrix_init
        else:
            img1= img2
    
