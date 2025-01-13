import eel
from io import BytesIO
from Data_processing import load_file, convert_rgb

eel.init("Frontend")  # initialising our directory
eel.start("index.html")


@eel.expose
def processing_image(image_file_array):
    file_1, image_1 = load_file(image_file_array[0])
    file_2, image_2 = load_file(image_file_array[0])
    rgb_image_1 = convert_rgb(file_1, image_1)
    rgb_image_2 = convert_rgb(file_2, image_2)

    # This (above) to CNN
    # Reconstructed image from CNN
    # i.e. from a function, returns what it needs to
    seg_image_one = breast_one.BytesIO()
    seg_image_two = breast_two.BytesIO()
    # processed_img.save(seg_image_one, 'JPEG') Saving it IN MEMORY , NOT DISK in case it is needed later
    seg_image_one.seek(
        0
    )  # Move the cursor to the front, this way it reads the file from the beginning
    seg_image_two.seek(0)
    processed_image_one = seg_image_one.getvalue()
    processed_image_two = seg_image_two.getvalue()

    # Return the byte stream (convert to base64 or directly send it)
    return processed_image_one, processed_image_two


# @eel.expose and then you define the function
# To call: eel.expose(function_in_java)
