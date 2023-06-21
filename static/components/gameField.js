class GameField extends HTMLElement {
    static numGameFields = 0;

    constructor () {
        super();
        GameField.numGameFields++;
    }

    connectedCallback () {
        this.innerHTML = 
        `<div>
            <button type="button" class="rem-pref">&#10005</button>
            <label for="game-title${GameField.numGameFields - 1}">Game title:</label>
            <input type="text" name="game-title" id="game-title${GameField.numGameFields - 1}" class="game-title">
            <label for="dislike-option${GameField.numGameFields - 1}">Dislike</label>
            <input type="radio" name="like-box${GameField.numGameFields - 1}" id="dislike-option${GameField.numGameFields - 1}" value="dislike">
            <label for="like-option${GameField.numGameFields - 1}">Like</label>
            <input type="radio" name="like-box${GameField.numGameFields - 1}" id="like-option${GameField.numGameFields - 1}" value="like">
            <div class="search-result"></div>
        </div>`;
    }

    disconnectedCallback () {
        GameField.numGameFields--;
    }
}

if ('customElements' in window) {
    customElements.define('game-field', GameField);
}