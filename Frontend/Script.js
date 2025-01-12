const menu = document.querySelector('#mobile-menu'); /*selects as a CSS selector, it returns the first item*/
const menuLinks = document.querySelector('.navbar__menu');
const upload_button=document.getElementById('upload_button');
const file_chosen=document.getElementById('file_chosen');
const button=document.querySelector('.main__button');

menu.addEventListener('click', function(){
    menu.classList.toggle('is-active'); /*classList is an object that represents the list of classes of an element*/
    menuLinks.classList.toggle('active');  /*add or removes active from the list of classes in menuLinks*/
    /*if .active is present then the menu links disappears */
});
let file_name = [];
upload_button.addEventListener('change', function () { //check here that two files are uploaded
    file_name.push(upload_button.files[0].name);
    file_name.push(upload_button.files[1].name);
    file_chosen.textContent = file_name.join(', ');
    button.classList.toggle('active');
});

let fileContent = [];

button.addEventListener('click', function () {
    fileContent = [];

    if (upload_button.files.length === 2) {
        fileContent.push(upload_button.files[0]);
        fileContent.push(upload_button.files[1])
        window.location.href = "Uploading_page.html"; //redirects page

        eel.processing_image(fileContent) (function(processed_image_one, processed_image_two) { //callback function
            // Set the processed image as the src of the image element
            document.getElementById('breast_one').src = processed_image_one;
            document.getElementById("breast_two").src = processed_image_two;
        })
    } else {
        alert("Please select two files.");
    }
});



/* eel.function_in_python(variable)
To define and expose function:
eel.expose(name_function)
function name_function (variable) {}
 */

/*
/!*Function that takes an image as input*!/
const button = document.querySelector('.main__button');
const imageInput = document.querySelector('#imageInput')
button.addEventListener("click", function() {
    imageInput.click();
    window.location.href = "Uploading_page.html";/!*put it after you are able to check the files are the one requested*!/
})

imageInput.addEventListener('change', function() {
    const files = imageInput.files;
    breast_one = files[0];
    breast_two= files[1];
})*/


