import Quadrant_densities
import cv2 as cv

import unittest
class test_quadrant_desnities(unittest.TestCase):

    def test_quadrant_densities(self):
        breast_mask_path = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\breast_masks\00a6b0d56eb5136c1be2c3d624b04dad.jpg"
        dense_mask_path = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\dense_masks\00a6b0d56eb5136c1be2c3d624b04dad.jpg"
        breastdensities = Quadrant_densities.quadrant_densities(breast_mask_path, dense_mask_path)
        self.assertEqual(breastdensities, [62.99891431098958, 50.41938730740444, 34.09023186141468, 57.68666190589542])

    def test_quadrant_densities_2(self):
        breast_mask_path1 = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\breast_masks\0a2290abc9c03b9eb387ddbfff12092c.jpg"
        dense_mask_path1 = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\dense_masks\0a2290abc9c03b9eb387ddbfff12092c.jpg"
        breastdensities1 = Quadrant_densities.quadrant_densities(breast_mask_path1, dense_mask_path1)
        self.assertEqual(breastdensities1, [14.82658487500104, 21.995611345081844, 17.04989194253012, 30.83032267237325])

    def test_quadrant_densities_flipped_image(self):
        picture_needs_flipping_dense = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\dense_masks\0a11ee9bbf17933d9680f74d64e02321.jpg"
        picture_needs_flipping_breast = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\breast_masks\0a11ee9bbf17933d9680f74d64e02321.jpg"
        flipped_image = Quadrant_densities.correct_orientation(picture_needs_flipping_dense)
        cv.imwrite("flipped_image.jpg", flipped_image)
        breastdensities2 = Quadrant_densities.quadrant_densities(picture_needs_flipping_breast, picture_needs_flipping_dense)
        self.assertEqual(breastdensities2, [17.995057684003964, 39.74046526349106, 48.85637993307146, 40.95893788955435])

if __name__ == '__main__':
    unittest.main()