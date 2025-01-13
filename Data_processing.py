# This file defines two functions:
# load_file reads a dicom file from the input path and extracts the pixel data into an array using pydicom
# convert_rgb checks the pixel value type using pydicom and converts it to RGB using numpy

import pydicom
from pydicom import pixel_array
from pydicom.pixel_data_handlers.util import convert_color_space
from pydicom.errors import InvalidDicomError
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_file(path):
    """
    Function reads a file from the input path(return file), checks if file type is DICOM,
    extracts the pixel data and inputs it into an array using pydicom(return image).
    Raises ValueError if pixel data is missing
    Raises TypeError if file is not DICOM
    :param path:
    :return file, image:
    """
    try:
        file = pydicom.dcmread(path)  # this reads the data
        if not hasattr(file, "PixelData") or not file.PixelData:
            raise ValueError("DICOM file is missing pixel data")
        image = file.pixel_array  # this converts pixel data into an array
        return (file, image)
    except InvalidDicomError:
        raise TypeError("The file uploaded is not a valid DICOM file")


def convert_rgb(file, image):
    """
    Function uses file reading and image pixel array data to check pixel value
    type and change it to RGB from MONOCHROME2 and YBR (the normal pixel value
    types in DICOM files), it returns RGB array of pixel values (rgb_image).
    Raises InvalidDicomError if photometric interpretation is not one of the described above
    Raises AttributeError if photometric interpretation value is missing
    :param file:
    :param image:
    :return rgb_image:
    """
    try:
        # for monochrome images
        if (
            file.PhotometricInterpretation == "MONOCHROME2"
        ):  # this checks the pixel type of the input
            image = (
                (image - image.min()) / (image.max() - image.min()) * 255
            )  # this normalizes the pixel values
            rgb_image = np.stack((image,) * 3, axis=-1)
            rgb_image = rgb_image.astype(np.uint8)  # this converts to uint8 type
        # for YBR images
        elif file.PhotometricInterpretation == "YBR_FULL_422":
            rgb_image = convert_color_space(
                image, current="YBR_FULL_422", desired="RGB"
            )  # this changes the pixel value type
        # add error for other color values? maybe not here?
        else:
            raise InvalidDicomError(
                f"Unsupported Photometric Interpretation: '{file.PhotometricInterpretation}'. "
                "Only 'MONOCHROME2' and 'YBR_FULL_422' are supported."
            )
        return rgb_image
    except AttributeError as e:
        raise AttributeError(
            "Invalid DICOM file structure. Missing Photometric Interpretation attribute"
        ) from e


if __name__ == "__main__":
    path = r"C:\Users\ihern\Documents\Java_try\breast_masking\image1.dcm"
    # file, image=load_file(path)
    # rgb_image=convert_rgb(file, image)
    # plt.imshow(rgb_image)
    # plt.show
    try:
        file, image = load_file(path)
        print("DICOM file loaded successfully!")
    except Exception as e:
        print(e)
