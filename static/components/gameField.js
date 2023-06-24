class GameField extends HTMLElement {
    static numGameFields = 0;

    constructor () {
        super();
        GameField.numGameFields++;
    }

    connectedCallback () {
        this.id_ = GameField.numGameFields - 1;
        this.gameId = null;
        this.pref = null;
        this.innerHTML = 
        `<div>
            <button type="button" class="rem-pref">&#10005</button>
            <label for="game-title${this.id_}">Game title:</label>
            <input type="text" name="game-title" id="game-title${this.id_}" class="game-title">
            <label for="dislike-option${this.id_}">Dislike</label>
            <input class="dislike-button" type="radio" name="like-box${this.id_}" id="dislike-option${this.id_}" value="dislike">
            <label for="like-option${this.id_}">Like</label>
            <input class="like-button" type="radio" name="like-box${this.id_}" id="like-option${this.id_}" value="like">
            <div class="search-result"></div>
        </div>`;
    }

    // disconnectedCallback () {
    //     GameField.numGameFields--;
    // }
}

if ('customElements' in window) {
    customElements.define('game-field', GameField);
}