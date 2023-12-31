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

const gameFields = {};
let recs = [];
let gettingRecs = false;

/**
 * Refresh the search results for known games in the model. 
 * @param {HTMLElement} gameField - The field that a new game is being searched in.
 * @returns null
 */
async function refreshSearch(gameField) {
    gameField.gameId = null;
    
    const gameTitle = gameField.getElementsByClassName("game-title").item(0);
    const searchContainer = gameField.getElementsByClassName("search-container").item(0);
    const searchResult = gameField.getElementsByClassName("search-result").item(0);

    let width = searchContainer.offsetWidth;
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

function addNewPrefField() {
    const gameField = document.createElement("game-field");
    // Must append before using children, need connectedCallback to run
    prefs.appendChild(gameField); 
    
    gameFields[gameField.id_] = gameField;
    const closeButton = gameField.getElementsByClassName("rem-pref").item(0);
    const dislike = gameField.getElementsByClassName("dislike-radio").item(0);
    const like = gameField.getElementsByClassName("like-radio").item(0);
    
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

    return gameField
}

/**
 * 
 * @param {*} event 
 * @returns 
 */
async function generateRecs(event) {
    if (event != null) {
        event.preventDefault();
    }
    
    if (gettingRecs == true) {
        return;
    }
    gettingRecs = true;
    
    if (Object.keys(gameFields).length == 0) {
        noPrefMsg.style.display = "block";
        gettingRecs = false;
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
            const top = gameField.offsetTop;

            gameTitle.classList.add("not-selected-warn");
            alert("Preference not selected. Please ensure all preferences are" + 
            " selected from the search list. This is indicated by title for" + 
            " preference having a blue outline");
            prefs.scroll({'top': top});

            gettingRecs = false;
            return;
        }

        let pref = gameField.pref;
        prefData[gameIndex] = pref;
    }

    // Load recs
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
    
    // Create rec elements
    const header = document.createElement("h2");
    header.style.display = "none";
    header.innerHTML = "Your Recommendations";
    recList.append(header);
    recs = [];
    for (const i in results) {
        const index = results[i][0];
        const id = String(results[i][1]);
        const title = results[i][2];
        
        const rec = document.createElement("li");
        rec.classList.add("rec");
        rec.style.display = "none";
        recs.push(rec);

        titleContainer = document.createElement("div");
        titleContainer.classList.add("title-container");
        
        // Create title
        const titleSpan = document.createElement("span");
        titleSpan.innerHTML = title;

        // Create like/dislike
        const prefButtons = document.createElement("pref-buttons");

        // Create image
        const response = await fetch("/get_cover/" + id);
        const data = await response.json();
        const img = new Image(width="100px");
        img.src = "data:image/jpg;base64," + data[0];
        img.classList.add("rec-image");
        
        titleContainer.appendChild(titleSpan);
        titleContainer.appendChild(prefButtons);
        rec.appendChild(titleContainer);
        rec.appendChild(img);
        recList.appendChild(rec);
        
        // Must access buttons after inserted into the DOM.
        const dislikeButt = prefButtons.getElementsByClassName("dislike-butt")[0];
        const likeButt = prefButtons.getElementsByClassName("like-butt")[0];

        dislikeButt.addEventListener("click", () => {
            gameField = addNewPrefField();
            dislikeRadio = gameField.getElementsByClassName("dislike-radio")[0];

            selectOption(gameField, index, title);
            gameField.pref = 0;
            dislikeRadio.checked = true;

            rec.remove()
        });
        likeButt.addEventListener("click", () => {
            gameField = addNewPrefField();
            selectOption(gameField, index, title);
            rec.remove()
        });
    }

    // Once done loading show recs
    loader.style.display = "none";
    loadMsg.style.display = "none";
    header.style.display = "block";
    for (const i in recs) {
        const rec = recs[i];
        rec.style.display = "flex";
    }
    
    gettingRecs = false;
}

addNewPref.addEventListener("click", addNewPrefField);
prefForm.addEventListener("submit", generateRecs);

clearAllButt.addEventListener("click", () => {
    if (confirm("Are you sure you want to clear your preferences?")) {
        // Clear prefs
        for (const i in gameFields) {
            let gameField = gameFields[i];
            gameField.remove();
            delete gameFields[i];
        }

        // Clear recs
        recList.innerHTML = '';
    }
});