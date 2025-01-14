# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 00:46:43 2025

@author: 27187
"""

import subprocess

def test_train_unet_hardcoded():
    """
    considerIng the determined path wrote in script, the test will directly test the script
    """
    
    script_path = r"F:\Leslie\uni\Y3\Programming\project\Repo\cancer_masking\CNN\newest_version\train_density_model_v1.py"  

    # call the script
    ret = subprocess.run(
        ["python", script_path],
        capture_output=True,   # capture stdout/stderr
        text=True              
    )

    # use pytest -s to get result
    print("STDOUT:", ret.stdout)
    print("STDERR:", ret.stderr)

    # check if script exit in the right code
    assert ret.returncode == 0, f"Script crashed with code {ret.returncode}"

    # check if the "training completed' after running
    assert "Training completed." in ret.stdout, "No 'Training completed.' found in script output'" 
