const SUCCESS_EMPTY = 204

const prefForm = document.getElementById("pref-form");
const gameTitle = document.getElementById("game-title");
const loadMsg = document.getElementById("load-msg");
const loader = document.getElementById("loader");
const noPrefMsg = document.getElementById("nopref-msg");
const recList = document.getElementById("rec-list");
const prefs = document.getElementById("prefs");

const addNewPref = document.getElementById("add-newpref");
const clearAllButt = document.getElementById("clear-all-butt");

const gameFields = {}

/**
 * Refresh the search results for known games in the model. 
 * @param {HTMLElement} gameField - The field that a new game is being searched in.
 * @returns null
 */
async function refreshSearch(gameField) {
    gameField.gameId = null;
    let width = gameField.offsetWidth;

    const gameTitle = gameField.getElementsByClassName("game-title").item(0);
    const searchResult = gameField.getElementsByClassName("search-result").item(0);

    searchResult.style.width = String(width) + "px"

    gameTitle.classList.remove("has-selected");

    const response = await fetch(`/search.${gameTitle.value}`);
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
    gameTitle.classList.add("has-selected");
    gameTitle.classList.remove("not-selected-warn");
}

/**
 * 
 * @param {*} event 
 * @returns 
 */
async function generateRecs(event) {
    event.preventDefault()

    if (Object.keys(gameFields).length == 0) {
        noPrefMsg.style.display = "block";
        return;
    } else {
        noPrefMsg.style.display = "none";
    }

    // Extract pref data from html elements
    const prefData = {};
    for (const i in gameFields) {
        const gameField = gameFields[i];

        gameIndex = gameField.gameId;
        if (gameIndex == null) {
            const gameTitle = gameField.getElementsByClassName("game-title").item(0);
            gameTitle.classList.add("not-selected-warn");
            return;
        }

        let pref = gameField.pref;
        prefData[gameIndex] = pref;
    }

    recList.innerHTML = "";

    loader.style.display = "block";
    loadMsg.style.display = "block";
    const response = await fetch("/recs", {
        method: "POST",
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(prefData)
    });

    const results = await response.json();
    loader.style.display = "none";
    loadMsg.style.display = "none";
    
    const header = document.createElement("h2");
    header.innerHTML = "Your Recommendations";
    recList.append(header);
    for (const i in results) {
        const title = results[i][1];
        const id = String(results[i][0]);
        console.log(title + " " + id);
        const rec = document.createElement("li");
        rec.classList.add("rec");
        rec.innerHTML = title;

        const response = await fetch("/get_cover/" + id);
        const data = await response.json();
        const img = new Image(width="100px");
        img.src = "data:image/jpg;base64," + data[0];
        img.classList.add("rec-image");
        
        rec.appendChild(img);
        recList.appendChild(rec);
    }
}

/**
 * Create new game entry field.
 */
addNewPref.addEventListener("click", () => {
    const gameField = document.createElement("game-field");
    // Must append before using children, need connectedCallback to run
    prefs.appendChild(gameField); 
    
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

prefForm.addEventListener("submit", generateRecs);
clearAllButt.addEventListener("click", () => {
    if (confirm("Are you sure you want to clear your preferences?") == true) {
        for (const i in gameFields) {
            let gameField = gameFields[i];
            gameField.remove();
            delete gameFields[i];
        }
    }
});