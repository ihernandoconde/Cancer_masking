import pydicom
import eel
from pydicom import pixel_array
from pydicom.pixel_data_handlers.util import convert_color_space
import numpy as np
from io import BytesIO

def load_file(path):
    #here or in the UI we should add a try catch to check if the file is actually dicom
    file=pydicom.dcmread(BytesIO(path.read()))  #this reads the data
    image= pixel_array(file)    #this converts pixel data into an array
    return file,image

def convert_rgb(file,image):
    #for monochrome images
    if file.PhotometricInterpretation=='MONOCHROME2':
        image=(image-image.min())/(image.max()-image.min())*255
        rgb_image=np.stack((image,)*3,axis=-1)
        rgb_image= rgb_image.astype(np.uint8)
    #for YBR images
    elif filePhotometricInterpretation=='YBR_FULL_422':
        rgb_image=convert_color_space(image, current="YBR_FULL_422", desired="RGB")
    #add error for other color values? maybe not here?
    else:
        rgb_image=None
        print('wtf')
    return rgb_image


