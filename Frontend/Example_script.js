// To contain all the Javascript files (makes the website interactive)
eel.interact_with_js ("Javascript");

eel.expose (interact_with_py)//I think they are already abstracted, so this makes them comunicate
function interact_with_py(){
    console.log("Hello from Python");
}