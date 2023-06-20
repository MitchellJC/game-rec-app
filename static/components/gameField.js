class GameField extends HTMLElement {
    static numGameFields = 0;

    constructor () {
        super();
        GameField.numGameFields++;
    }

    connectedCallback () {
        this.innerHTML = 
        `<div>
            <button type="button" class="rem-pref">-</button>
            <label for="game-title${GameField.numGameFields + 1}">Game title:</label>
            <input type="text" name="game-title" id="game-title${GameField.numGameFields + 1}">
            <label for="dislike-option${GameField.numGameFields + 1}">Dislike</label>
            <input type="radio" name="like-box" id="dislike-option${GameField.numGameFields + 1}" value="dislike">
            <label for="like-option${GameField.numGameFields + 1}">Like</label>
            <input type="radio" name="like-box" id="like-option${GameField.numGameFields + 1}" value="like">
        </div>`;
    }

    disconnectedCallback () {
        GameField.numGameFields--;
    }
}

if ('customElements' in window) {
    customElements.define('game-field', GameField);
}