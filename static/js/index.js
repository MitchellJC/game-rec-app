const SUCCESS_EMPTY = 204

const prefForm = document.getElementById("pref-form");
const gameTitle = document.getElementById("game-title");
const addNewPref = document.getElementById("add-newpref");

const prefs = {}

addNewPref.addEventListener("click", () => {
    const gameField = document.createElement("game-field");
    prefForm.appendChild(gameField); // Must append before accessing children -- connectedCallback

    const closeButtons = gameField.getElementsByClassName("rem-pref");
    for (const i in closeButtons) {
        const button = closeButtons.item(i);
        button.addEventListener("click", () => {
            gameField.remove()
        })
    }

    gameField.addEventListener("keyup", () => {
        refreshSearch(gameField);
    });
})

async function refreshSearch(gameField) {
    const title = gameField.getElementsByClassName("game-title").item(0).value;
    const searchResult = gameField.getElementsByClassName("search-result").item(0);

    const response = await fetch(`/search.${title}`);
    const status = await response.status;

    searchResult.innerHTML = ""
    if (status == SUCCESS_EMPTY) {
        return;
    }
    const results = await response.json();

    // Update search result options
    for (const i in results) {
        let title, id;
        [title, id] = results[i];
        const prefNum = Object.keys(prefs).length
        const searchOption = document.createElement("button");

        searchOption.addEventListener("click", () => selectOption(gameField, prefNum, id, title));
        searchOption.innerHTML = `${title}, ${id}`;
        searchResult.appendChild(searchOption);
    }
}

function selectOption(gameField, prefNum, id, title) {
    const gameTitle = gameField.getElementsByClassName("game-title").item(0);
    const searchResult = gameField.getElementsByClassName("search-result").item(0);

    searchResult.innerHTML = "";
    gameTitle.value = title;
    prefs[prefNum] = {'id': id, 'title': title};
}