class PrefButtons extends HTMLElement {
    constructor () {
        super();
    }

    connectedCallback() {
        this.innerHTML = 
        `<div class="pref-buttons">
            <button class="dislike-butt" type="button">Dislike</button>
            <button class="like-butt" type="button">Like</button>
        </div>`;
    }

}

if ('customElements' in window) {
    customElements.define('pref-buttons', PrefButtons);
}