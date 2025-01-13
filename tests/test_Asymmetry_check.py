import Quadrant_densities
from Asymmetry_check import quadrant_asymmetry_check
from Asymmetry_check import general_asymmetry_check
import unittest


class test_asymmetry_check(unittest.TestCase):

    def test_quadrant_densities_same_breast(self):
        breast_mask_path = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\breast_masks\00a6b0d56eb5136c1be2c3d624b04dad.jpg"
        dense_mask_path = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\dense_masks\00a6b0d56eb5136c1be2c3d624b04dad.jpg"
        breastdensities = Quadrant_densities.quadrant_densities(breast_mask_path, dense_mask_path)

        breast_mask_path1 = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\breast_masks\0a2290abc9c03b9eb387ddbfff12092c.jpg"
        dense_mask_path1 = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\dense_masks\0a2290abc9c03b9eb387ddbfff12092c.jpg"
        breastdensities1 = Quadrant_densities.quadrant_densities(breast_mask_path1, dense_mask_path1)

        symmetry = quadrant_asymmetry_check(breastdensities, breastdensities)
        self.assertEqual(symmetry, [False, False, False, False])

    def test_quadrant_densities_different_breasts(self):
        breast_mask_path = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\breast_masks\00a6b0d56eb5136c1be2c3d624b04dad.jpg"
        dense_mask_path = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\dense_masks\00a6b0d56eb5136c1be2c3d624b04dad.jpg"
        breastdensities = Quadrant_densities.quadrant_densities(breast_mask_path, dense_mask_path)

        breast_mask_path1 = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\breast_masks\0a2290abc9c03b9eb387ddbfff12092c.jpg"
        dense_mask_path1 = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\dense_masks\0a2290abc9c03b9eb387ddbfff12092c.jpg"
        breastdensities1 = Quadrant_densities.quadrant_densities(breast_mask_path1, dense_mask_path1)

        symmetry1 = quadrant_asymmetry_check(breastdensities, breastdensities1)
        self.assertEqual(symmetry1, [True, True, True, True])

    def test_densities_at_threshold(self):
        symmetry3 = general_asymmetry_check(50, 60)
        self.assertTrue(symmetry3 == False)

    def test_densities_at_threshold_swapped(self):
        symmetry4 = general_asymmetry_check(60, 50)
        self.assertTrue(symmetry4 == False)

    def test_densities_same(self):
        symmetry5 = general_asymmetry_check(50, 50)
        self.assertTrue(symmetry5 == False)

    def test_densities_assymetric(self):
        symmetry6 = general_asymmetry_check(61, 50)
        self.assertTrue(symmetry6 == True)

if __name__ == '__main__':
    unittest.main()
