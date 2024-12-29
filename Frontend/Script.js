const menu = document.querySelector('#mobile-menu'); /*selects as a CSS selector, it returns the first item*/
const menuLinks = document.querySelector('.navbar__menu');
const upload_button=document.getElementById('upload_button');
const file_name=document.getElementById('file_chosen')

menu.addEventListener('click', function(){
    menu.classList.toggle('is-active'); /*classList is an object that represents the list of classes of an element*/
    menuLinks.classList.toggle('active');  /*add or removes active from the list of classes in menuLinks*/
    /*if .active is present then the menu links disappears */
});

upload_button.addEventListener('change', function () {
    file_name.textContent = this.files[0].name

});
function doSomething(){
    var fileContents = document.getElementById('upload_button').value;

}