import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space
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

#path=r"C:\Users\ihern\Documents\Java_try\breast_masking\image1.dcm"
#file, image=load_file(path)
#rgb_image=convert_rgb(file, image)
#plt.imshow(rgb_image)
#plt.show
