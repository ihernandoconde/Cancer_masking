import numpy as np
import cv2 as cv
"""
Purpose: to split the breast into 4 regions (quadrants) and find the individual densities of each region. Identifying 
which regions are more dense can help inform medical professionals on which regions are more likely to contain cancer, 
if it is present. 
"""
def correct_orientation (image):
    """
    Images of the right breast can stay as they are, but images of the left breast need to flipped vertically to ensure
    that the labelling of the quadrants is consistent between the two.
    :param image (of breast mask or density mask)
    t:return: an image that is flipped (if necessary) so that the nipple is on the left hand side
    """
    if isinstance(image, str):
        image = cv.imread(image, cv.IMREAD_GRAYSCALE)

    height, width = image.shape
    right_side  = image[:, width//2:]
    left_side = image[:, :width//2]
    if np.sum(left_side) > np.sum(right_side):
        image = cv.flip(image, 1) # flip the image vertically if the left side has more pixels than the right side
    return image

def background_mask (breast_mask, dense_mask):
    """
    The breast region is outlined and this outline is applied to the dense mask to isolate the region of interest,
    so that when the image is split into quadrants, the background of the image is not taken into account.
    :param breast_mask: outlines the breast region
    :param dense_mask: shows the dense regions in the breast
    :return: breast region mask and the dense image (with the mask applied)
    """
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
    """
    To calculate the density of each quadrant, the breast region is split into four regions (quadrants), and the density
    of each quadrant is calculated.
    :param breast_mask: outlines the breast region
    :param dense_mask: shows the dense regions of the breast
    :return: density percentages for each quadrant of the breast
    """
    breast_region_mask, masked_dense = background_mask(breast_mask, dense_mask)

    #find box of breast region:
    y, x = np.where(breast_region_mask>0)
    miny, maxy = y.min(), y.max()
    min_x, max_x = x.min(), x.max()

    #crop to the bounding region
    cropped_breast = breast_region_mask[miny:maxy, min_x:max_x]
    cropped_dense = masked_dense[miny:maxy, min_x:max_x]

    height, width = cropped_breast.shape #dense.shape return dimensions of the array

    quadrants = [ #dividing image into 4 quadrants
        (0, height//2, width//2, width), # (rows from top to middle, columns from middle to right edge)
        (0, height//2, 0, width//2),
        (height//2, height, width//2, width),
        (height//2, height, 0, width//2)
    ]

    quadrant_densities = []
    for y_start, y_end, x_start, x_end in quadrants: # for each of the 4 quadrants
        breastquadrant = cropped_breast[y_start:y_end, x_start:x_end] #slices the image array to get the quadrant portion
        densequadrant = cropped_dense[y_start:y_end, x_start:x_end]

        breastpixelcount = np.sum(breastquadrant>0) #total area of breast
        densepixelcount = np.sum(densequadrant>128) #total dense area of breast

        if breastpixelcount ==0:
            density = 0
        else:
            density = (densepixelcount / breastpixelcount)*100
        quadrant_densities.append(density)

    return quadrant_densities
