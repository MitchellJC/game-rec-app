class GameField extends HTMLElement {
    static numGameFields = 0;

    constructor () {
        super();
        GameField.numGameFields++;
    }

    connectedCallback () {
        this.id_ = GameField.numGameFields - 1;
        this.gameId = null;
        this.pref = 1;
        this.innerHTML = 
        `<div class="game-field">
            <button type="button" class="rem-pref">&#10005</button>
            <label for="game-title${this.id_}">Game title:</label>
            <input type="text" name="game-title" id="game-title${this.id_}" class="game-title" required>
            <label for="dislike-option${this.id_}">Dislike</label>
            <input class="dislike-radio" type="radio" name="like-box${this.id_}" id="dislike-option${this.id_}" value="dislike">
            <label for="like-option${this.id_}">Like</label>
            <input class="like-radio" type="radio" name="like-box${this.id_}" id="like-option${this.id_}" value="like" checked>
            <div class="search-result"></div>
        </div>`;
    }

}

if ('customElements' in window) {
    customElements.define('game-field', GameField);
}