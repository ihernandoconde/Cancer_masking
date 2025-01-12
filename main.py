import eel
from Data_processing import loadfile, convertrgb
from deploy import load_image, predict_masks, save_results, compute_density_ensemble
from qudrantdensities import background_removal, quadrant_densities

eel.init("Frontend") #initialising our directory

@eel.expose
def get_results(uploaded_files): #uploaded_files will be considered a list since we can have 1 or 2 files

    for uploaded_file in uploaded_files:
        file, image = loadfile(uploaded_file)
        rgb_image = convertrgb(file, image) 
        breast_pred, dense_pred = predict_masks(rgb_image, model)
        breast_path, dense_path = save_results(rgb_image, breast_pred, dense_pred, save_dir)
        final_density = compute_density_ensemble(breast_mask_arr, dense_mask_arr, linreg_model, rf_model, catboost_model, threshold=128)
        #Some elements needed in deploy are outside the function: gotta figure out if i should 
        # copy the lines or if there is another way to incorporate it.
        masked_dense = background_removal(breast_path, dense_path)
        quadrant_densities = quadrant_densities(breast_path, dense_path)
        result =  f"Breast density: {final_density}, Quadrant densities: {quadrant_densities}"
    
    return result






images = eel.get_files() #file content from Java
eel.expose(get_results)

print(images)
eel.start('index.html')
eel.start('Uploading_page.html')#AT THE END!!
eel.start('Home_page.html')

#@eel.expose and then you define the function
#To call: eel.expose(function_in_java)
