#This file defines two functions:
#load_file reads a dicom file from the input path and extracts the pixel data into an array using pydicom
#convert_rgb checks the pixel value type using pydicom and converts it to RGB using numpy

import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space
from pydicom.errors import InvalidDicomError
import matplotlib.pyplot as plt
import numpy as np


def load_file(path):
    #here or in the UI we should add a try catch to check if the file is actually dicom
    file=pydicom.dcmread(path)  #this reads the data
    image= file.pixel_array   #this converts pixel data into an array
    return (file,image)
def convert_rgb(file,image):
    #for monochrome images
    if file.PhotometricInterpretation=='MONOCHROME2':
        image=(image-image.min())/(image.max()-image.min())*255
        rgb_image=np.stack((image,)*3,axis=-1)
        rgb_image= rgb_image.astype(np.uint8)
    #for YBR images
    elif file.PhotometricInterpretation=='YBR_FULL_422':
        rgb_image=convert_color_space(image, current="YBR_FULL_422", desired="RGB")
    #add error for other color values? maybe not here?
    else:
        print('wtf')
    return(rgb_image)

try:
    file, image = load_file(path)
    print("DICOM file loaded successfully!")
except Exception as e:
    print(e)
