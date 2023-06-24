const SUCCESS_EMPTY = 204

const prefForm = document.getElementById("pref-form");
const gameTitle = document.getElementById("game-title");
const addNewPref = document.getElementById("add-newpref");
const submitButton = document.getElementById("submit-button");
const loadMsg = document.getElementById("load-msg");
const recList = document.getElementById("rec-list");

const gameFields = {}

/**
 * Refresh the search results for known games in the model. 
 * @param {HTMLElement} gameField - The field that a new game is being searched in.
 * @returns null
 */
async function refreshSearch(gameField) {
    const title = gameField.getElementsByClassName("game-title").item(0).value;
    const searchResult = gameField.getElementsByClassName("search-result").item(0);

    const response = await fetch(`/search.${title}`);
    const status = response.status;

    searchResult.innerHTML = ""
    if (status == SUCCESS_EMPTY) {
        return;
    }
    const results = await response.json();

    // Update search result options
    for (const i in results) {
        let title, id;
        [title, id] = results[i];
        const searchOption = document.createElement("button");

        searchOption.addEventListener("click", () => selectOption(gameField,
             id, title));
        searchOption.innerHTML = `${title}, ${id}`;
        searchResult.appendChild(searchOption);
    }
}

/**
 * Select the given search result on the given gameField. 
 * @param {HTMLElement} gameField 
 * @param {number} prefNum 
 * @param {number} id 
 * @param {string} title 
 */
function selectOption(gameField, id, title) {
    const gameTitle = gameField.getElementsByClassName("game-title").item(0);
    const searchResult = gameField.getElementsByClassName("search-result").item(0);
    gameField.gameId = id;

    searchResult.innerHTML = "";
    gameTitle.value = title;
}

/**
 * 
 */
async function generateRecs() {
    // Extract pref data from html elements
    const prefData = {};
    for (const i in gameFields) {
        const gameField = gameFields[i];
        gameIndex = gameField.gameId;
        pref = gameField.pref;

        prefData[gameIndex] = pref;
    }
    loadMsg.style.visibility = "visible";
    const response = await fetch("/recs", {
        method: "POST",
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(prefData)
    });

    const results = await response.json();
    console.log(results)
    loadMsg.style.visibility = "hidden";

    for (const i in results) {
        const title = results[i][1];
        const rec = document.createElement("li");
        rec.innerHTML = title;
        recList.appendChild(rec);
    }
}

/**
 * Create new game entry field.
 */
addNewPref.addEventListener("click", () => {
    const gameField = document.createElement("game-field");
    // Must append before using children, need connectedCallback to run
    prefForm.appendChild(gameField); 
    
    gameFields[gameField.id_] = gameField;
    const closeButton = gameField.getElementsByClassName("rem-pref").item(0);
    const dislike = gameField.getElementsByClassName("dislike-button").item(0);
    const like = gameField.getElementsByClassName("like-button").item(0);
    
    closeButton.addEventListener("click", () => {
        gameField.remove();
        delete gameFields[gameField.id_]; 
    });
    
    gameField.addEventListener("keyup", () => {
        refreshSearch(gameField);
    });

    gameField.addEventListener("click", () => {
        if (dislike.checked) {
            gameField.pref = 0;
        } else if (like.checked) {
            gameField.pref = 1;
        }
    });
});

submitButton.addEventListener("click", generateRecs);