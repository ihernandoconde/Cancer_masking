import eel # The python is the backend
eel.init("Frontend")
eel.start('Page 1.html', block=False)

@eel.expose #so things from the javaScript side are exposed into the
#python side (as they are two separate languages)
def interact_with_js (x):
    print(x)
#don't know how to make it comunicate with javaScrip
eel.interact_with_py()