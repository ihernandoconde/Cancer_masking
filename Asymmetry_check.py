from pathlib import Path

"""
Asymmetry between breasts can be a risk factor of breast cancer and asymmetry between quadrants can indicate which areas
are more at high risk.
"""
def general_asymmetry_check(density1, density2):
    """
    If there is more than 10% difference in the density of 2 breasts, they can be classed as asymmetric.
    Args:
        density1: Density of breast 1 (float)
        density2: Density of breast 2 (float)

    Returns: true if asymmetric, false otherwise

    """
    if abs(density1 - density2) > 10:
        return True
    else:
        return False

def quadrant_asymmetry_check(quadrant_density1, quadrant_density2):
    """
    To localise where the asymmetry is, it can be checked for each quadrant.
    Args:
        quadrant_density1: list of floats representing density of 4 quadrants for breast 1
        quadrant_density2: list of floats representing density of 4 quadrants for breast 2

    Returns: list of boolean where true shows asymmetry, false shows no asymmetry.
    """
    quadrant_asymmetry_list = []
    for i in range (0,4):
        if abs(quadrant_density1[i] - quadrant_density2[i]) > 10:
            quadrant_asymmetry_list.append(True)
        else:
            quadrant_asymmetry_list.append(False)
    return quadrant_asymmetry_list


