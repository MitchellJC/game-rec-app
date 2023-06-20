const prefForm = document.getElementById("pref-form");
const searchResult = document.getElementById("search-result");
const gameTitle = document.getElementById("game-title");

prefForm.addEventListener("keyup", () => {
    var title = gameTitle.value;

    fetch(`/search.${title}`).then((response) => {
        return response.text()
    }).then((text)=> {
        searchResult.innerHTML = text;
    });

});

function refreshSearch(text) {
    console.log(text);
}