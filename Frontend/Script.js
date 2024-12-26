const menu = document.querySelector('#mobile-menu'); /*selects as a CSS selector, it returns the first item*/
const menuLinks = document.querySelectorAll('.navbar__links');

menu.addEventListener('click', () => {
    menu.classList.toggle('is.active'); /*classList is an object that represents the list of classes of an element*/
    menuLinks.classList.toggle('active');  /*add or removes active from the list of classes in menuLinks*/
    /*if .active is present then the menu links disappears */
})
