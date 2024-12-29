import eel
from io import BytesIO
from Functions import *

eel.init("Frontend") #initialising our directory
eel.start('index.html')
files = eel.get_files() #file content from Java

file_1, image_1=load_file(files[0])
file_2, image_2=load_file(files[0])

rgb_image_1 = convert_rgb(file_1, image_1)
rgb_image_2 = convert_rgb(file_2, image_2)

print(files)
# eel.start('index.html')
eel.start('Uploading_page.html')#AT THE END!!
eel.start('Home_page.html')

#@eel.expose and then you define the function
#To call: eel.expose(function_in_java)