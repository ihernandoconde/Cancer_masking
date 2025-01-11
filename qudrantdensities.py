
from PIL import Image
import numpy as np
import cv2 as cv


def background_clean (breast_mask, dense_mask):
    # This function 'cleans' the image so that there are no white bits/ artifacts outside the breast region.
    # This is useful so that when the image is cropped in the next function, it isnt affected by white pixels that are not in the breast region
    breast = cv.imread(breast_mask, cv.IMREAD_GRAYSCALE) #load image in greyscale
    dense = cv.imread(dense_mask, cv.IMREAD_GRAYSCALE)
    _, breast_binary_mask = cv.threshold(breast, 128, 255, cv.THRESH_BINARY) #converts greyscale mask into binary mask
    #pixels with intensity > 128 are set to 255 (white). intensity <128 set to 0 (black)
    contours, _ = cv.findContours(breast_binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #contours of breast mask
    #finds boundaries of all white regions in the breast_binary_mask
    largest_contour = max(contours, key=cv.contourArea) #contour with largest area
    breast_region_mask = np.zeros_like(breast_binary_mask) #empty binary mask with same dimensions as breast_binary_mask
    cv.drawContours(breast_region_mask, [largest_contour], -1, 255, thickness=cv.FILLED) #drawing  largest contour on mask
    masked_dense = cv.bitwise_and(dense, dense, mask=breast_region_mask) #masking the dense image to region inside contour

    cv.imshow("Dense Mask", masked_dense)  # Display the binary mask
    cv.waitKey(0)
    cv.destroyAllWindows()

    return breast_region_mask, masked_dense

def quadrant_densities(breast_mask, dense_mask):
    #remove the background to get the ROI
    breast_region_mask, masked_dense = background_clean(breast_mask, dense_mask)

    #find box of breast region:
    y, x = np.where(breast_region_mask>0)
    miny, maxy = y.min(), y.max()
    min_x, max_x = x.min(), x.max()

    #crop to the bounding region
    cropped_breast = breast_region_mask[miny:maxy, min_x:max_x]
    cropped_dense = masked_dense[miny:maxy, min_x:max_x]

    #breast = cv.bitwise_and(cropped_breast, cropped_breast, mask = resized_mask)
    #dense = cv.bitwise_and(cropped_dense, cropped_dense, mask = resized_mask)

    small_breast = cv.resize(cropped_breast, (256, 256))
    cv.imshow("Cropped breast", small_breast)  # Display the binary mask
    cv.waitKey(0)
    cv.destroyAllWindows()
    small_dense = cv.resize(cropped_dense, (256, 256))
    cv.imshow("Cropped dense", small_dense)  # Display the binary mask
    cv.waitKey(0)
    cv.destroyAllWindows()

    height, width = cropped_breast.shape #dense.shape return dimensions of the array
    quadrants = [ #dividing image into 4 quadrants
        (0, height//2, width//2, width), #top right (rows from top to middle, columns from middle to right edge)
        (0, height//2, 0, width//2), #top left
        (height//2, height, width//2, width), #bottom right
        (height//2, height, 0, width//2) #bottom left
    ]

    quadrant_densities = []
    for y_start, y_end, x_start, x_end in quadrants: # for each of the 4 quadrants
        breastquadrant = cropped_breast[y_start:y_end, x_start:x_end] #slices the image array to get the qudrant portion
        densequadrant = cropped_dense[y_start:y_end, x_start:x_end]

        breastpixelcount = np.sum(breastquadrant>0) #total area of breast
        densepixelcount = np.sum(densequadrant>128) #total dense area of breat

        if breastpixelcount ==0:
            density = 0
        else:
            density = (densepixelcount / breastpixelcount)*100
        quadrant_densities.append(density)

    #display image with correct aspect ratio (copied from chatgpt)
    height, width = cropped_breast.shape
    scale_factor = min(800 / height, 800 / width)  # Choose the smaller scale to preserve aspect ratio
    resized_dense = cv.resize(cropped_dense, (int(width * scale_factor), int(height * scale_factor)))
    # Display the resized image
    cv.imshow("Breast with Quadrants", resized_dense)
    cv.waitKey(0)
    cv.destroyAllWindows()

    percentages = [f"{density:.2f}%" for density in quadrant_densities] #turns the numbers in percentages to decimal places
    return percentages



breast_mask_path = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\breast_masks\00a6b0d56eb5136c1be2c3d624b04dad.jpg"
dense_mask_path = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\dense_masks\00a6b0d56eb5136c1be2c3d624b04dad.jpg"
breastdensities = quadrant_densities(breast_mask_path, dense_mask_path)
print(f"Quadrant densities: {breastdensities}")

