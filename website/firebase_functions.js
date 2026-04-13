// ============================================
// IMPORTS
// ============================================
import "https://www.gstatic.com/firebasejs/10.7.1/firebase-app-compat.js";
import "https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore-compat.js";
import "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth-compat.js";
import "https://www.gstatic.com/firebasejs/10.7.1/firebase-storage-compat.js";
import { firebaseConfig, siteConfig } from "./firebase_config.js";

// ============================================
// FIREBASE CONFIGURATION
// ============================================

let db;
let auth;
try {
    firebase.initializeApp(firebaseConfig);
    db = firebase.firestore();
    auth = firebase.auth();
    console.log("Firebase initialized successfully");

// Set subtitle from siteConfig
if (siteConfig && siteConfig.ownerName) {
    document.getElementById("site-subtitle").textContent =
        `Live Bird Sightings from ${siteConfig.ownerName}'s Backyard`;
}
} catch (error) {
    console.error("Error initializing Firebase:", error);
    showError("Failed to connect to Firebase. Please check your configuration.");
}

// ============================================
// PAGINATION STATE
// ============================================
const PAGE_SIZE = 8;
let currentPage = 1;
let _cachedSightings = [];

// ============================================
// ADMIN AUTHENTICATION
// ============================================

let isAdmin = false;
let currentUser = null;

auth.onAuthStateChanged((user) => {
    if (user) {
        currentUser = user;
        isAdmin = true;
        showAdminStatus();
        console.log("Admin logged in:", user.email);
    } else {
        currentUser = null;
        isAdmin = false;
        hideAdminStatus();
        console.log("No user logged in");
    }
    if (db) {
        loadSightings();
    }
});

function showAdminStatus() {
    document.getElementById('admin-login-btn').style.display = 'none';
    document.getElementById('admin-status').style.display = 'flex';
    document.getElementById('actions-column-header').style.display = 'table-cell';
    if (currentUser) {
        const username = currentUser.email.split('@')[0];
        document.getElementById('admin-username').textContent = `Logged in as ${username}`;
    }
}

function hideAdminStatus() {
    document.getElementById('admin-login-btn').style.display = 'block';
    document.getElementById('admin-status').style.display = 'none';
    document.getElementById('actions-column-header').style.display = 'none';
}

function showLoginModal() {
    document.getElementById('login-modal').style.display = 'flex';
    document.getElementById('username').focus();
}

function hideLoginModal() {
    document.getElementById('login-modal').style.display = 'none';
    document.getElementById('login-form').reset();
    document.getElementById('login-error').style.display = 'none';
}

function showLoginError(message) {
    const errorDiv = document.getElementById('login-error');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

async function handleLogin(username, password) {
    try {
        const email = `${username}@birdnerd.local`;
        await auth.signInWithEmailAndPassword(email, password);
        hideLoginModal();
    } catch (error) {
        console.error("Login error:", error);
        if (error.code === 'auth/user-not-found' || error.code === 'auth/wrong-password') {
            showLoginError('Invalid username or password');
        } else if (error.code === 'auth/too-many-requests') {
            showLoginError('Too many failed attempts. Please try again later.');
        } else {
            showLoginError('Login failed. Please try again.');
        }
    }
}

async function handleLogout() {
    if (confirm('Are you sure you want to log out?')) {
        try {
            await auth.signOut();
            console.log("Logged out successfully");
        } catch (error) {
            console.error("Logout error:", error);
            alert("Error logging out. Please try again.");
        }
    }
}

async function deleteSighting(sightingId, birdName) {
    if (!isAdmin) {
        alert("You must be logged in as admin to delete sightings.");
        return;
    }

    if (confirm(`Are you sure you want to delete this sighting?\n\n${birdName}\n\nThis action cannot be undone.`)) {
        try {
            await db.collection('sightings').doc(sightingId).delete();
            console.log(`Deleted Firestore doc: ${sightingId}`);

            try {
                const storagePath = `sightings/${sightingId}.jpg`;
                const storageRef = firebase.storage().ref(storagePath);
                await storageRef.delete();
                console.log(`Deleted Storage image: ${storagePath}`);
            } catch (storageErr) {
                if (storageErr.code !== 'storage/object-not-found') {
                    console.warn("Storage image delete warning:", storageErr.message);
                }
            }
        } catch (error) {
            console.error("Error deleting sighting:", error);
            alert("Failed to delete sighting. Please try again.");
        }
    }
}

// Event listeners for login/logout
document.getElementById('admin-login-btn').addEventListener('click', showLoginModal);
document.getElementById('admin-logout-btn').addEventListener('click', handleLogout);
document.getElementById('cancel-login').addEventListener('click', hideLoginModal);

document.getElementById('login-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value;
    handleLogin(username, password);
});

document.getElementById('login-modal').addEventListener('click', (e) => {
    if (e.target.id === 'login-modal') hideLoginModal();
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.getElementById('login-modal').style.display === 'flex') {
        hideLoginModal();
    }
});

// ============================================
// ABOUT MODAL
// ============================================

document.getElementById('about-btn').addEventListener('click', () => {
    document.getElementById('about-modal').style.display = 'flex';
});

document.getElementById('close-about').addEventListener('click', () => {
    document.getElementById('about-modal').style.display = 'none';
});

document.getElementById('about-modal').addEventListener('click', (e) => {
    if (e.target.id === 'about-modal') {
        document.getElementById('about-modal').style.display = 'none';
    }
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.getElementById('about-modal').style.display === 'flex') {
        document.getElementById('about-modal').style.display = 'none';
    }
});

// ============================================
// PI HEARTBEAT INDICATOR
// ============================================
// main.py writes a heartbeat document to Firestore:
//   collection: 'heartbeat', document: 'status'
//   fields: { alive: bool, timestamp: Firestore server timestamp }
//
// The indicator turns green if alive === true AND the timestamp
// is less than HEARTBEAT_TIMEOUT_MS old. Otherwise red.
// A setInterval re-evaluates the already-received timestamp every
// 30 seconds so the dot goes red if the Pi silently stops sending.

const HEARTBEAT_TIMEOUT_MS = 3 * 60 * 1000; // 3 minutes

let _lastHeartbeat = null; // { alive: bool, timestamp: Date }

function updatePiIndicator() {
    const dot   = document.getElementById('pi-dot');
    const label = document.getElementById('pi-label');

    if (!_lastHeartbeat) {
        dot.className   = 'pi-dot pi-unknown';
        label.textContent = 'Pi: Unknown';
        return;
    }

    const age       = Date.now() - _lastHeartbeat.timestamp.getTime();
    const fresh     = age < HEARTBEAT_TIMEOUT_MS;
    const connected = _lastHeartbeat.alive && fresh;

    if (connected) {
        dot.className     = 'pi-dot pi-connected';
        label.textContent = 'Pi: Online';
    } else {
        dot.className     = 'pi-dot pi-disconnected';
        label.textContent = 'Pi: Offline';
    }
}

function listenHeartbeat() {
    if (!db) return;

    db.collection('heartbeat').doc('status').onSnapshot((doc) => {
        if (doc.exists) {
            const data = doc.data();
            const ts   = data.timestamp && data.timestamp.toDate
                ? data.timestamp.toDate()
                : null;
            if (ts) {
                _lastHeartbeat = { alive: data.alive === true, timestamp: ts };
                updatePiIndicator();
            }
        } else {
            // Document doesn't exist yet (placeholder state)
            _lastHeartbeat = null;
            updatePiIndicator();
        }
    }, (err) => {
        console.warn("Heartbeat listener error:", err);
    });

    // Re-evaluate age every 30 seconds so the dot goes red if Pi goes silent
    setInterval(updatePiIndicator, 30_000);
}

// ============================================
// BIRD SOUND + MUTE
// ============================================

let _soundEnabled  = true;   // sound is always available from page load
let _muted         = false;
let _initialLoad   = true;   // suppress sound on first data load
let _knownIds      = new Set();

const _chirpAudio = document.getElementById('bird-chirp');

function playBirdSound() {
    if (_muted || !_soundEnabled) return;
    _chirpAudio.currentTime = 0;
    _chirpAudio.play().catch(() => {
        // Autoplay blocked - browser requires a user gesture first.
        // The sound will work after the user clicks anything on the page.
    });
}

function showMuteButton() {
    document.getElementById('mute-btn').style.display = 'inline-flex';
}

document.getElementById('mute-btn').addEventListener('click', () => {
    _muted = !_muted;
    document.getElementById('mute-btn').textContent = _muted ? 'Bird sounds: OFF' : 'Bird sounds: ON';
});

// Show mute button immediately on page load
showMuteButton();

// ============================================
// UTILITY FUNCTIONS
// ============================================

function formatTimestamp(timestamp, timezone) {
    let date;
    if (timestamp && timestamp.toDate) {
        date = timestamp.toDate();
    } else if (timestamp instanceof Date) {
        date = timestamp;
    } else {
        return "Unknown";
    }

    // Fall back to America/New_York if timezone is an offset like "UTC-4"
    // which browsers don't accept as a valid IANA timezone
    const safeTimezone = (timezone && !timezone.startsWith('UTC'))
        ? timezone
        : 'America/New_York';

    const options = {
        weekday: 'long',
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        timeZone: safeTimezone,
        timeZoneName: 'short'
    };
    return date.toLocaleString('en-US', options);
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.90) return 'confidence-high';
    if (confidence >= 0.70) return 'confidence-medium';
    return 'confidence-low';
}

// Format confidence to 2 decimal places.
// Capped at 99.99 - a model prediction of 100% is never academically honest.
// Floored at 0.01 so secondary predictions never display as 0.00%.
function formatConfidence(confidence) {
    const percent = confidence * 100;
    const capped  = Math.min(percent, 99.99);
    const floored = Math.max(capped, 0.01);
    return parseFloat(floored.toFixed(2));
}

function formatTopPredictions(predictions) {
    if (!predictions || predictions.length === 0) return predictions;
    return predictions.map(pred => ({
        ...pred,
        confidence: formatConfidence(pred.confidence)
    }));
}

function showError(message) {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('error').style.display = 'block';
    document.getElementById('error-message').textContent = message;
}

function hideError() {
    document.getElementById('error').style.display = 'none';
}

function getRelativeTime(timestamp) {
    let date;
    if (timestamp && timestamp.toDate) {
        date = timestamp.toDate();
    } else if (timestamp instanceof Date) {
        date = timestamp;
    } else {
        return "Unknown";
    }

    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return formatTimestamp(timestamp, 'America/New_York').split(',')[0];
}

// ============================================
// PAGINATION
// ============================================

function getTotalPages() {
    return Math.max(1, Math.ceil(_cachedSightings.length / PAGE_SIZE));
}

function goToPage(page) {
    const total = getTotalPages();
    currentPage = Math.max(1, Math.min(page, total));
    renderSightings(_cachedSightings);
    document.getElementById('table-container').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderPagination(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const total = getTotalPages();
    const current = currentPage;

    if (total <= 1) {
        container.innerHTML = '';
        return;
    }

    const pages = new Set();
    pages.add(1);
    pages.add(total);
    for (let p = Math.max(1, current - 1); p <= Math.min(total, current + 1); p++) {
        pages.add(p);
    }
    const pageList = [...pages].sort((a, b) => a - b);

    let html = `<div class="pagination">`;
    html += `<button class="page-btn page-prev" ${current === 1 ? 'disabled' : ''} data-page="${current - 1}">&#8592; Prev</button>`;

    let prev = null;
    for (const p of pageList) {
        if (prev !== null && p - prev > 1) {
            html += `<span class="page-ellipsis">&hellip;</span>`;
        }
        html += `<button class="page-btn ${p === current ? 'page-active' : ''}" data-page="${p}">${p}</button>`;
        prev = p;
    }

    html += `<button class="page-btn page-next" ${current === total ? 'disabled' : ''} data-page="${current + 1}">Next &#8594;</button>`;

    const startEntry = (current - 1) * PAGE_SIZE + 1;
    const endEntry = Math.min(current * PAGE_SIZE, _cachedSightings.length);
    html += `<span class="page-info">Showing ${startEntry}–${endEntry} of ${_cachedSightings.length}</span>`;
    html += `</div>`;

    container.innerHTML = html;

    container.querySelectorAll('.page-btn:not([disabled])').forEach(btn => {
        btn.addEventListener('click', () => goToPage(parseInt(btn.dataset.page)));
    });
}

// ============================================
// DATA LOADING AND DISPLAY
// ============================================

function updateStatistics(sightings) {
    document.getElementById('total-sightings').textContent = sightings.length;

    const uniqueSpecies = new Set(sightings.map(s => s.scientific_name));
    document.getElementById('unique-species').textContent = uniqueSpecies.size;

    // Count how many sightings have a timestamp from today in the local timezone
    const today = new Date();
    const seenToday = sightings.filter(s => {
        if (!s.timestamp || !s.timestamp.toDate) return false;
        const sightingDate = s.timestamp.toDate();
        return sightingDate.getFullYear() === today.getFullYear() &&
               sightingDate.getMonth() === today.getMonth() &&
               sightingDate.getDate() === today.getDate();
    }).length;
    document.getElementById('seen-today').textContent = seenToday;

    if (sightings.length > 0) {
        const lastTime = getRelativeTime(sightings[0].timestamp);
        document.getElementById('last-sighting').textContent = lastTime;
    } else {
        document.getElementById('last-sighting').textContent = "None yet";
    }
}

// ============================================
// BIRD DETAIL MODAL
// ============================================

let _allSightings = [];

function openBirdModal(sighting) {
    const scientificName = sighting.scientific_name;
    const now = new Date();

    const allTime = _allSightings.filter(s => s.scientific_name === scientificName).length;
    const last30  = _allSightings.filter(s => {
        const d = s.timestamp && s.timestamp.toDate ? s.timestamp.toDate() : null;
        return d && s.scientific_name === scientificName && (now - d) <= 30 * 86400000;
    }).length;
    const last7   = _allSightings.filter(s => {
        const d = s.timestamp && s.timestamp.toDate ? s.timestamp.toDate() : null;
        return d && s.scientific_name === scientificName && (now - d) <= 7 * 86400000;
    }).length;

    const modalImg   = document.getElementById('bird-modal-img');
    const modalGif   = document.getElementById('bird-modal-gif');
    const noImgText  = document.getElementById('bird-modal-no-img');
    const dlWrap     = document.getElementById('bird-modal-download-wrap');
    const dlBtn      = document.getElementById('bird-modal-download-btn');

    if (sighting.gif_url) {
        modalGif.src = sighting.gif_url;
        modalGif.style.display = 'block';
        modalImg.style.display = 'none';
        noImgText.style.display = 'none';

        // Show download button and wire it up
        dlWrap.style.display = 'block';
        dlBtn.onclick = () => downloadGif(sighting.gif_url, sighting.timestamp);

    } else if (sighting.image_url) {
        modalImg.src = sighting.image_url;
        modalImg.style.display = 'block';
        modalGif.style.display = 'none';
        noImgText.style.display = 'none';
        dlWrap.style.display = 'none';
    } else {
        modalImg.style.display = 'none';
        modalGif.style.display = 'none';
        noImgText.style.display = 'block';
        dlWrap.style.display = 'none';
    }

    document.getElementById('bird-modal-common').textContent     = sighting.common_name;
    document.getElementById('bird-modal-scientific').textContent  = sighting.scientific_name;
    document.getElementById('bird-modal-time').textContent        = formatTimestamp(sighting.timestamp, sighting.timezone);
    document.getElementById('bird-modal-confidence').textContent  = formatConfidence(sighting.confidence) + '%';
    document.getElementById('bird-modal-confidence').className    = 'confidence ' + getConfidenceClass(sighting.confidence);

    document.getElementById('bird-modal-alltime').textContent = allTime;
    document.getElementById('bird-modal-30days').textContent  = last30;
    document.getElementById('bird-modal-7days').textContent   = last7;

    const ebirdLink = document.getElementById('bird-modal-ebird');
    ebirdLink.href = 'https://ebird.org/home';
    ebirdLink.textContent = `Learn more about ${sighting.common_name} on eBird`;

    document.getElementById('bird-modal').style.display = 'flex';
}

function closeBirdModal() {
    document.getElementById('bird-modal').style.display = 'none';
    document.getElementById('bird-modal-img').src = '';
    document.getElementById('bird-modal-gif').src = '';
}

// ============================================
// GIF DOWNLOAD
// ============================================

async function downloadGif(gifUrl, timestamp) {
    try {
        const response = await fetch(gifUrl);
        const blob = await response.blob();
        const url  = URL.createObjectURL(blob);

        // Build filename: bird-nerd-YYYY-MM-DD_HH-MM-SS.gif
        let datePart = 'unknown';
        if (timestamp && timestamp.toDate) {
            const d = timestamp.toDate();
            const pad = n => String(n).padStart(2, '0');
            datePart = `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}`
                     + `_${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}`;
        }
        const filename = `bird-nerd-${datePart}.gif`;

        const a = document.createElement('a');
        a.href     = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    } catch (err) {
        console.error("GIF download failed:", err);
        alert("Could not download GIF. Try right-clicking the image and saving manually.");
    }
}

// ============================================
// MODAL EVENT LISTENERS
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('bird-modal').addEventListener('click', (e) => {
        if (e.target.id === 'bird-modal') closeBirdModal();
    });
    document.querySelector('.bird-modal-close').addEventListener('click', closeBirdModal);
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.getElementById('bird-modal').style.display === 'flex') {
        closeBirdModal();
    }
});

// ============================================
// TABLE RENDERING
// ============================================

const IMAGE_PREVIEW_HEIGHT_PX = 200;

function renderSightings(sightings) {
    const tbody = document.getElementById('sightings-body');
    tbody.innerHTML = '';

    const start = (currentPage - 1) * PAGE_SIZE;
    const pageSightings = sightings.slice(start, start + PAGE_SIZE);

    if (sightings.length === 0) {
        const colspan = isAdmin ? 5 : 4;
        tbody.innerHTML = `<tr><td colspan="${colspan}" class="no-data">No bird sightings yet. Check back soon!</td></tr>`;
        renderPagination('pagination-top');
        renderPagination('pagination-bottom');
        return;
    }

    pageSightings.forEach(sighting => {
        const row = document.createElement('tr');
        row.style.cursor = 'pointer';
        row.title = 'Click to view details';
        row.addEventListener('click', (e) => {
            if (e.target.classList.contains('delete-btn')) return;
            openBirdModal(sighting);
        });

        // Date & Time
        const dateCell = document.createElement('td');
        const fullTimestamp = formatTimestamp(sighting.timestamp, sighting.timezone);
        const parts = fullTimestamp.split(', ');
        if (parts.length >= 3) {
            const datePart = parts.slice(0, -1).join(', ');
            const timePart = parts[parts.length - 1];
            dateCell.innerHTML = `${datePart}<br>${timePart}`;
        } else {
            dateCell.textContent = fullTimestamp;
        }
        row.appendChild(dateCell);

        // Bird Species
        const speciesCell = document.createElement('td');
        speciesCell.innerHTML = `
          <div>${sighting.common_name}</div>
          <div class="scientific-name">${sighting.scientific_name}</div>
        `;
        row.appendChild(speciesCell);

        // Confidence
        const confidenceCell = document.createElement('td');
        const confidencePercent = formatConfidence(sighting.confidence);
        confidenceCell.innerHTML = `<span class="confidence ${getConfidenceClass(sighting.confidence)}">${confidencePercent}%</span>`;
        row.appendChild(confidenceCell);

        // Image Preview
        const imageCell = document.createElement('td');
        imageCell.className = 'image-preview-cell';
        if (sighting.gif_url) {
            const gif = document.createElement('img');
            gif.src = sighting.gif_url;
            gif.alt = sighting.common_name;
            gif.className = 'sighting-thumbnail';
            gif.style.height = IMAGE_PREVIEW_HEIGHT_PX + 'px';
            gif.loading = 'lazy';
            imageCell.appendChild(gif);
        } else if (sighting.image_url) {
            const img = document.createElement('img');
            img.src = sighting.image_url;
            img.alt = sighting.common_name;
            img.className = 'sighting-thumbnail';
            img.style.height = IMAGE_PREVIEW_HEIGHT_PX + 'px';
            img.loading = 'lazy';
            imageCell.appendChild(img);
        } else {
            const noImg = document.createElement('span');
            noImg.className = 'no-image-text';
            noImg.textContent = 'No image available :(';
            imageCell.appendChild(noImg);
        }
        row.appendChild(imageCell);

        // Admin Actions
        if (isAdmin) {
            const actionsCell = document.createElement('td');
            actionsCell.className = 'actions-cell';
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'delete-btn';
            deleteBtn.textContent = 'Delete';
            deleteBtn.onclick = (e) => {
                e.stopPropagation();
                deleteSighting(sighting.id, sighting.common_name);
            };
            actionsCell.appendChild(deleteBtn);
            row.appendChild(actionsCell);
        }

        tbody.appendChild(row);
    });

    renderPagination('pagination-top');
    renderPagination('pagination-bottom');
}

function loadSightings() {
    hideError();

    db.collection('sightings')
        .orderBy('timestamp', 'desc')
        .onSnapshot((snapshot) => {
            const sightings = [];
            snapshot.forEach((doc) => {
                const data = doc.data();
                data.id = doc.id;
                sightings.push(data);
            });

            console.log(`Loaded ${sightings.length} sightings`);

            // Detect genuinely new sightings (not the initial page load)
            if (_initialLoad) {
                // Populate known IDs silently on first load
                sightings.forEach(s => _knownIds.add(s.id));
                _initialLoad = false;
            } else {
                // Any ID we haven't seen before is a new arrival
                const newOnes = sightings.filter(s => !_knownIds.has(s.id));
                if (newOnes.length > 0) {
                    newOnes.forEach(s => _knownIds.add(s.id));
                    playBirdSound();
                }
            }

            _allSightings  = sightings;
            _cachedSightings = sightings;
            currentPage = 1;

            document.getElementById('loading').style.display = 'none';
            document.getElementById('table-container').style.display = 'block';

            updateStatistics(sightings);
            renderSightings(sightings);

        }, (error) => {
            console.error("Error loading sightings:", error);
            showError(`Error loading data: ${error.message}`);
        });
}

// ============================================
// INITIALIZE
// ============================================

if (db) {
    loadSightings();
    listenHeartbeat();
}