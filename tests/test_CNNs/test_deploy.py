# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 00:51:30 2025

@author: 27187
"""

import subprocess
import os


def test_inference_script():
    """
    only check if the code could work under real env, considering the path wrote in script
    """
    env = os.environ.copy()#use env including all required packages
    env["PYTHONPATH"] = r"F:\Leslie\uni\Y3\Programming\project\Repo\cancer_masking\CNN"
    
    python_exe = "python"

    script_path = r"F:\Leslie\uni\Y3\Programming\project\Repo\cancer_masking\CNN\newest_version\deploy_v2.py"

    # use subprocess to test the code
    ret = subprocess.run(
        [python_exe, script_path],   
        env=env,                     
        capture_output=True,
        text=True
    )

    # print subprocess stdout/stderr (pytest -s)
    print("STDOUT:", ret.stdout)
    print("STDERR:", ret.stderr)

    # check if the code exit in right code
    assert ret.returncode == 0, f"Script crashed with code {ret.returncode}"

    # final checks base on script's output
    assert "Segmentation model loaded successfully." in ret.stdout, \
        "Did not load segmentation model as expected"
    assert "Final Combined Density" in ret.stdout, \
        "No 'Final Combined Density' message found in script output"

