import Quadrant_densities
from Asymmetry_check import quadrant_asymmetry_check
from Asymmetry_check import general_asymmetry_check

breast_mask_path = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\breast_masks\00a6b0d56eb5136c1be2c3d624b04dad.jpg"
dense_mask_path = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\dense_masks\00a6b0d56eb5136c1be2c3d624b04dad.jpg"
breastdensities = Quadrant_densities.quadrant_densities(breast_mask_path, dense_mask_path)
print(breastdensities)

breast_mask_path1 = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\breast_masks\0a2290abc9c03b9eb387ddbfff12092c.jpg"
dense_mask_path1 = r"C:\Users\aizaa\OneDrive\Documents\Imperial College London\biomedical engineering\year 3\software engineering\project\Mammogram Density Assessment Dataset\main_dataset\main_dataset\train\dense_masks\0a2290abc9c03b9eb387ddbfff12092c.jpg"
breastdensities1 = Quadrant_densities.quadrant_densities(breast_mask_path1, dense_mask_path1)
print(breastdensities1)

symmetry = quadrant_asymmetry_check(breastdensities, breastdensities)
symmetry1 = quadrant_asymmetry_check(breastdensities, breastdensities1)
print(symmetry) # should expect false, false, false, false because the breast is being compared to itself
print(symmetry1)

symmetry3 = general_asymmetry_check(50, 60)
print(symmetry3)
symmetry4 = general_asymmetry_check(60, 50)
print(symmetry4)
symmetry5 = general_asymmetry_check(50, 50)
print(symmetry5)
symmetry6 = general_asymmetry_check(61, 50)
print(symmetry6)