import eel
eel.init("Frontend") #initialising our directory

images = eel.get_files() #file content from Java


print(images)
eel.start('index.html')
eel.start('Uploading_page.html')#AT THE END!!
eel.start('Home_page.html')

#@eel.expose and then you define the function
#To call: eel.expose(function_in_java)