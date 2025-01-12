import numpy as np
import cv2 as cv

def correct_orientation (image):
    #To ensure consistency of quadrant labelling, images are flipped so that the nipple is on the left hand side if not already
    height, width = image.shape
    right_side  = image[:, width//2:]
    left_side = image[:, :width//2]
    if np.sum(left_side) > np.sum(right_side):
        image = cv.flip(image, 1) # flip the image vertically if the left side has more pixels than the right side
    return image

def background_clean (breast_mask, dense_mask):
    #'cleans' the image so that there are no white areas outside the breast region, so that the breast region can be isolated accurately.
    breast = cv.imread(breast_mask, cv.IMREAD_GRAYSCALE) #load image in greyscale
    dense = cv.imread(dense_mask, cv.IMREAD_GRAYSCALE)
    breast =  correct_orientation(breast)
    dense = correct_orientation(dense)
    _, breast_binary_mask = cv.threshold(breast, 128, 255, cv.THRESH_BINARY) #converts greyscale mask into binary mask
    #pixels with intensity > 128 are set to 255 (white). intensity <128 set to 0 (black)
    contours, _ = cv.findContours(breast_binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #finds boundaries of all white regions in the breast_binary_mask
    largest_contour = max(contours, key=cv.contourArea) #contour with largest area
    breast_region_mask = np.zeros_like(breast_binary_mask) #empty binary mask
    cv.drawContours(breast_region_mask, [largest_contour], -1, 255, thickness=cv.FILLED) #drawing  largest contour on mask
    masked_dense = cv.bitwise_and(dense, dense, mask=breast_region_mask) #masking the dense image to region inside contour

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

    height, width = cropped_breast.shape #dense.shape return dimensions of the array

    quadrant_labels = [
        "Top right quadrant:",
        "Top left quadrant:",
        "Bottom right quadrant:",
        "Bottom left quadrant:",
    ]
    quadrants = [ #dividing image into 4 quadrants
        (0, height//2, width//2, width), # (rows from top to middle, columns from middle to right edge)
        (0, height//2, 0, width//2),
        (height//2, height, width//2, width),
        (height//2, height, 0, width//2)
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

    percentages = [f"{label}{density:.2f}%" for label, density in zip(quadrant_labels,quadrant_densities)]
    return percentages

def display_image(cropped_dense):
    # display smaller image with correct aspect ratio
    height, width = cropped_dense.shape
    scale_factor = min(800 / height, 800 / width)
    small_dense = cv.resize(cropped_dense, (int(width * scale_factor), int(height * scale_factor)))
    cv.imshow("Dense mask", small_dense)
    cv.waitKey(0)
    cv.destroyAllWindows()


breast_mask_path = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\breast_masks\00a6b0d56eb5136c1be2c3d624b04dad.jpg"
dense_mask_path = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\dense_masks\00a6b0d56eb5136c1be2c3d624b04dad.jpg"
breastdensities = quadrant_densities(breast_mask_path, dense_mask_path)
print(f"Quadrant densities: {breastdensities}")
