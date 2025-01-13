import unittest
from unittest.mock import Mock, patch
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom import examples
from Data_processing import convert_rgb, load_file
import numpy as np


class test_data_processing(unittest.TestCase):

    def test_load_valid_file(self):
        #Create a valid file
        mock_path=examples.get_path("rgb_color")
        #mock_file.PixelData = b"\x00\x01\x02"  # Mock pixel data
        #mock_file.pixel_array = np.array([[0, 1], [2, 3]])  # Mock pixel array

        file, image=load_file(mock_path)
        self.assertIsNotNone(file)
        self.assertTrue((image==pydicom.pixel_array(file)).all())

    def test_load_non_dicom_file(self):
        with patch("pydicom.dcmread", side_effect=InvalidDicomError):
            with self.assertRaises(TypeError):
                load_file("fake_path.dcm")

    def test_load_no_pixel_data_file(self):

        #Create mock file with no pixel data
        mock_file = Mock(spec=pydicom.dataset.FileDataset)
        del mock_file.PixelData
        with patch("pydicom.dcmread",return_value=mock_file):
            with self.assertRaises(ValueError):
                load_file(mock_file)

        
    #def test_convert_monochrome(self):

    #def test_convert_ybr(self):

    def test_convert_unsupported_type(self):
        #create file with different photometric interpretation
        mock_path=examples.get_path("rgb_color")
        file, image=load_file(mock_path)
        file.PhotometricInterpretation="INVALID_TYPE"

        with self.assertRaises(InvalidDicomError):
            convert_rgb(file, image)


    def convert_no_photometric_interpretation_file(self):
        #create file with no photometric interpretation
        mock_path=examples.get_path("rgb_color")
        file, image=load_file(mock_path)
        del file.PhotometricInterpretation

        with self.assertRaises(AttributeError):
            convert_rgb(file, image)


if __name__ == '__main__':
    unittest.main(verbosity=2)
