class PrefButtons extends HTMLElement {
    constructor () {
        super();
    }

    connectedCallback() {
        this.innerHTML = 
        `<div class="pref-buttons">
            <button type="button">Dislike</button>
            <button type="button">Like</button>
        </div>`;
    }

}

if ('customElements' in window) {
    customElements.define('pref-buttons', PrefButtons);
}