import cv2
from PIL import Image
import numpy as np
import cv2 as cv
#import csv
def background_removal (breast_mask, dense_mask):
    breast = cv.imread(breast_mask, cv.IMREAD_GRAYSCALE) #load image in greyscale
    dense = cv.imread(dense_mask, cv.IMREAD_GRAYSCALE)
    print("Min and Max values in breast image:", np.min(breast), np.max(breast))
    _, breast_binary_mask = cv.threshold(breast, 128, 255, cv.THRESH_BINARY) #threshold to make binary image
    cv.imshow("Binary Mask", breast_binary_mask)  # Display the binary mask
    cv.waitKey(0)
    cv.destroyAllWindows()
    contours, _ = cv.findContours(breast_binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #contours of breast mask
    breast_region_mask = np.zeros_like(breast_binary_mask) #empty mask for breast region
    cv.imshow("Contours on Mask", breast_region_mask)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.drawContours(breast_region_mask, contours, -1, (255), thickness=cv.FILLED) #drawing contours on mask
    masked_dense = cv.bitwise_and(dense, dense, mask=breast_region_mask) #masking the dense image to region inside contour

    cv.imshow('Masked Dense', masked_dense)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return masked_dense


def quadrant_densities(breast_mask, dense_mask):
    breastmask = Image.open(breast_mask).convert("L") #convert("L") puts image in greyscale mode
    densemask = Image.open(dense_mask).convert("L")

    breastmask = breastmask.resize((256, 256))
    densemask = densemask.resize((256, 256))
    # resizing so that all images are the same size but this might reduce precision and distort details of some images

    breast = np.array(breastmask) #numpy array
    dense = np.array(densemask)


    threshold = 128 #there is a clear contrast between breast tissue and backgroudn in breast mask, so fixed threshold should be sufficient
    masked_breast = (breast > threshold).astype(np.uint8) #binarizing the breast mask
    # Binarizing is when you convert image to a binary format- where pixels are either 0(background) or 1(foreground)
    # pixels with intesnity greater than threshold are foreground (breast) and those with less are considered background
    # this removes the background of the image so we only focus on breast region
    masked_dense = dense * masked_breast #applying mask to dense image
    height, width = masked_dense.shape #dense.shape return dimensions of the array
    quadrants = [ #dividing image into 4 quadrants
        (0, height//2, width//2, width), #top right (rows from top to middle, columns from middle to right edge)
        (0, height//2, 0, width//2), #top left
        (height//2, height, width//2, width), #bottom right
        (height//2, height, 0, width//2) #bottom left
    ]

    quadrant_densities = []
    for y_start, y_end, x_start, x_end in quadrants: # for each of the 4 quadrants
        breastquadrant = masked_breast[y_start:y_end, x_start:x_end] #slices the image array to get the qudrant portion
        densequadrant = masked_dense[y_start:y_end, x_start:x_end]

        breastpixelcount = np.sum(breastquadrant>0) #total area of breast
        densepixelcount = np.sum(densequadrant>threshold) #total dense area of breat

        if breastpixelcount ==0:
            density = 0
        else:
            density = (densepixelcount / breastpixelcount)*100
        quadrant_densities.append(density)

    percentages = [f"{density:.2f}%" for density in quadrant_densities] #turns the numbers in percentages to decimal places
    return percentages

#to check that images are processed properly, cv2 can be used to display the images:
def image_check(breast_mask, dense_mask):
    breastmask = Image.open(breast_mask).convert("L")
    densemask = Image.open(dense_mask).convert("L")
    breast = np.array(breastmask)
    dense = np.array(densemask)

    threshold = 128  # there is a clear contrast between breast tissue and backgroudn in breast mask, so fixed threshold should be sufficient
    masked_breast = (breast > threshold).astype(np.uint8)  # binarizing the breast mask
    masked_dense = masked_breast * dense
    smaller_breastimage = cv.resize(masked_breast, (256,256))
    smaller_denseimage = cv.resize(masked_dense, (256,256))

    height, width = smaller_denseimage.shape  # dense.shape return dimensions of the array
    quadrants = [  # dividing image into 4 quadrants
        (0, height // 2, width // 2, width),  # top right (rows from top to middle, columns from middle to right edge)
        (0, height // 2, 0, width // 2),  # top left
        (height // 2, height, width // 2, width),  # bottom right
        (height // 2, height, 0, width // 2)  # bottom left
    ] #quadrant is a list with 4 tuples, each tuple has 4 values (y_start/top, y_end/bottom, x_start/left, x_end/right)

    for i in range (4): #looping through the 4 quadrants
        top, bottom, left, right = quadrants[i] #coordiantes of quadrant
        breast_quadrant = smaller_breastimage[top:bottom, left:right]
        dense_mask = smaller_denseimage[top:bottom, left:right]

        cv.imshow(f'Breast Quadrant {i+1}', breast_quadrant*255)
        cv.imshow(f'Dense Quadrant {i+1}', dense_mask*255)
    cv.waitKey(0)
    cv.destroyAllWindows()



breast_mask_path = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\breast_masks\00a6b0d56eb5136c1be2c3d624b04dad.jpg"
dense_mask_path = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\dense_masks\00a6b0d56eb5136c1be2c3d624b04dad.jpg"
breastdensities = quadrant_densities(breast_mask_path, dense_mask_path)
print(f"Quadrant densities: {breastdensities}")
#image_check(breast_mask_path, dense_mask_path)
background_removal(breast_mask_path, dense_mask_path)