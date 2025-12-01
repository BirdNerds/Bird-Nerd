// Fetches JSON from /api/photos and renders a responsive gallery


async function loadGallery() {
    try {
        const resp = await fetch('/api/photos');
        const items = await resp.json();


        const container = document.getElementById('gallery');
        const tmpl = document.getElementById('card-template');


        if (!Array.isArray(items) || items.length === 0) {
            container.innerHTML = '<p>No photos found.</p>';
            return;
        }


        for (const it of items) {
            const clone = tmpl.content.cloneNode(true);
            const img = clone.querySelector('img');
            img.src = it.url;
            img.alt = it.species || it.filename;


            clone.querySelector('.species').textContent = it.species || 'Unknown species';
            clone.querySelector('.datetime').textContent = (it.date && it.time) ? `${it.date} ${it.time}` : '';
            clone.querySelector('.notes').textContent = it.notes || '';


            // lightbox: click to open full-size
            img.addEventListener('click', () => window.open(it.url, '_blank'));


            container.appendChild(clone);
        }
    } catch (e) {
        console.error(e);
        document.getElementById('gallery').innerHTML = '<p>Error loading gallery.</p>';
    }
}


window.addEventListener('DOMContentLoaded', loadGallery);