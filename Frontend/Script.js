const menu = document.querySelector('#mobile-menu'); /*selects as a CSS selector, it returns the first item*/
const menuLinks = document.querySelector('.navbar__menu');
const upload_button=document.getElementById('upload_button');
const file_chosen=document.getElementById('file_chosen');
const button=document.querySelector('.main__button');
const label2=document.getElementById('label2');

/*Making the menu links disappear or appear*/
menu.addEventListener('click', function(){
    menu.classList.toggle('is-active'); /*classList is an object that represents the list of classes of an element*/
    menuLinks.classList.toggle('active');  /*add or removes active from the list of classes in menuLinks*/
});


/*The upload button used to send collect the files uploaded by the user for the processing.
It displays the name of the files selected */
let file_name = [];
upload_button.addEventListener('change', function () { //check here that two files are uploaded
    if (upload_button.files.length  ===1) {
        file_name.push(upload_button.files[0].name);
        file_chosen.textContent = this.files[0].name;
        label2.classList.toggle('active');
    }
    if (upload_button.files.length  ===2) {
        file_name.push(upload_button.files[0].name);
        file_name.push(upload_button.files[1].name);
        file_chosen.textContent = this.files[0].name+ this.files[1].name;
        label2.classList.toggle('active');
    }
    button.classList.toggle('active');
});


/* A function that contains all the different explanations of the BI-RADS category. Goes on the page
where their category is revealed.*/
function density_explanation(density) {
     if (density <25) {
               paragraph_content = "<p>Category A: The breast is mostly fatty. This means any abnormalities" +
                   " (such as cancer) can be easily seen on the mammogram. If the radiologist didn’t notice " +
                   "anything out of ordinary, then there probably isn’t. </p>"
           }
     else if (density >=25 && density <50) {
               paragraph_content = "<p>Category B: Scattered density. The breast has a lot of fatty tissue, " +
                   "and a few areas of dense tissue scattered about. This is still considered low-density" +
                   " as abnormalities will still be visible in the mammogram.</p>"
               }
     else if (density >=50 && density<75) {
         paragraph_content= "<p>Category C: Heterogeneously dense. This means that, while there is mostly dense" +
             " tissue, it is not clustered together but spread around. This could obscure small abnormal masses," +
             " if any is present, making it harder for the mammogram to show them. It is therefore extremely " +
             "important for you to bring this to your doctor’s attention and discuss it with them. \n Remember, this" +
             " in of itself does not signify that you may have cancer. This density assessment is used for the same" +
             " purpose as taking a mammogram: to check for any abnormalities and, if there are, detect them early" +
             " and treat them. </p>"
     }
     else {
         paragraph_content= "<p>Category D: Extremely dense. Only 10% of women have been in this category. The " +
             "breast is mostly made up of dense tissue. This lowers the sensitivity of mammography: the mammogram " +
             "may not be good enough in showing any masses of cancer (if present), as they are masked by the dense" +
             " tissue. It is therefore extremely important for you to bring this to your doctor’s attention and " +
             "discuss it with them. \n Remember, this in of itself does not signify that you may have cancer." +
             " This density assessment is used for the same purpose as taking a mammogram: to check for any " +
             "abnormalities and, if there are, detect them early and treat them.\n </p>"
     }
}


/*After uploading the image, the user is redirected to another page (Uploading_page) where:
-   the segmented images will be displayed
-   the BI-RADS category of the user will be revealed
 */
let paragraph_content = "";
let fileContent = [];

button.addEventListener('click', function () {
    fileContent = []; //To clear previous elements that may have been selected

    if (upload_button.files.length  ===1) {
        fileContent.push(upload_button.files[0]);
        window.location.href = "Uploading_page.html";//redirects page

        eel.processing_image(fileContent) (function(processed_image_one, processed_image_two, density_one, density_two) { //callback function
            document.getElementById('breast_one').src = processed_image_one; // Sets the processed image as the source of the image element
            density_explanation(density_one);
            document.getElementById("birads_explanation").innerHTML = paragraph_content;
        })
    }
    else if (upload_button.files.length  ===2) {
        fileContent.push(upload_button.files[0]);
        fileContent.push(upload_button.files[1]);
        window.location.href = "Uploading_page.html";

        eel.processing_image(fileContent) (function(processed_image_one, processed_image_two, density_one, density_two) { //callback function
            document.getElementById('breast_one').src = processed_image_one;
            document.getElementById("breast_two").src = processed_image_two;
            density_explanation(density_one);
            document.getElementById("birads_explanation").innerHTML = paragraph_content;
            density_explanation(density_two);
            document.getElementById("birads_explanation_two").innerHTML = paragraph_content;
        })
    }
    else if(upload_button.files.length <1) {
        alert("Please select at least one file.");
    }
    else {
        alert("Please select a maximum of two files.");
    }
});

eel.expose(get_files);
function get_files(){
    return fileContent;
}




