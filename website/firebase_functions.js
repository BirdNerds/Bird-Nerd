// ============================================
// IMPORTS
// ============================================
import "https://www.gstatic.com/firebasejs/10.7.1/firebase-app-compat.js";
import "https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore-compat.js";
import "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth-compat.js";

// ============================================
// FIREBASE CONFIGURATION
// ============================================
// TODO: Replace this with YOUR Firebase config from the Firebase Console
// You got this when you registered your web app in Step 5 of the setup
const firebaseConfig = {
    apiKey: "AIzaSyBwq3mq6rS-3wMykWvV-oZl1wuh40esqIo",
    authDomain: "bird-nerd-27eb1.firebaseapp.com",
    projectId: "bird-nerd-27eb1",
    storageBucket: "bird-nerd-27eb1.firebasestorage.app",
    messagingSenderId: "832918617928",
    appId: "1:832918617928:web:fe79157c19fa944823bd7d"
};

// Initialize Firebase
let db;
let auth;
try {
    firebase.initializeApp(firebaseConfig);
    db = firebase.firestore();
    auth = firebase.auth();
    console.log("Firebase initialized successfully");
} catch (error) {
    console.error("Error initializing Firebase:", error);
    showError("Failed to connect to Firebase. Please check your configuration.");
}

// ============================================
// ADMIN AUTHENTICATION
// ============================================

let isAdmin = false;
let currentUser = null;

// Monitor auth state
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
    // Reload sightings to show/hide delete buttons
    if (db) {
        loadSightings();
    }
});

function showAdminStatus() {
    document.getElementById('admin-login-btn').style.display = 'none';
    document.getElementById('admin-status').style.display = 'flex';
    document.getElementById('actions-column-header').style.display = 'table-cell';
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
        // Convert username to email format for Firebase Auth
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
            console.log(`Deleted sighting: ${sightingId}`);
            // The onSnapshot listener will automatically update the UI
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

// Close modal when clicking outside
document.getElementById('login-modal').addEventListener('click', (e) => {
    if (e.target.id === 'login-modal') {
        hideLoginModal();
    }
});

// Close modal with Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.getElementById('login-modal').style.display === 'flex') {
        hideLoginModal();
    }
});

// ============================================
// UTILITY FUNCTIONS
// ============================================

function formatTimestamp(timestamp, timezone) {
    // Handle Firestore Timestamp object
    let date;
    if (timestamp && timestamp.toDate) {
        date = timestamp.toDate();
    } else if (timestamp instanceof Date) {
        date = timestamp;
    } else {
        return "Unknown";
    }

    // Format with the timezone from the database
    const options = {
        weekday: 'long', // Shows "Monday", "Tuesday", etc.
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        timeZone: timezone || 'America/New_York', // Use stored timezone or default to EST
        timeZoneName: 'short' // Shows "EST", "EDT", etc.
    };
    return date.toLocaleString('en-US', options);
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.90) return 'confidence-high'; // 90% - 100% = high (green)
    if (confidence >= 0.70) return 'confidence-medium'; // 70% - 90% = medium (orange)
    return 'confidence-low'; // 0% - 70% = low (red)
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
    return formatTimestamp(timestamp, 'America/New_York').split(',')[0]; // Just the date
}

// ============================================
// DATA LOADING AND DISPLAY
// ============================================

function updateStatistics(sightings) {
    // Total sightings
    document.getElementById('total-sightings').textContent = sightings.length;

    // Unique species
    const uniqueSpecies = new Set(sightings.map(s => s.scientific_name));
    document.getElementById('unique-species').textContent = uniqueSpecies.size;

    // Last sighting
    if (sightings.length > 0) {
        const lastTime = getRelativeTime(sightings[0].timestamp);
        document.getElementById('last-sighting').textContent = lastTime;
    } else {
        document.getElementById('last-sighting').textContent = "None yet";
    }
}

function renderSightings(sightings) {
    const tbody = document.getElementById('sightings-body');
    tbody.innerHTML = ''; // Clear existing rows

    if (sightings.length === 0) {
        const colspan = isAdmin ? 5 : 4;
        tbody.innerHTML = `<tr><td colspan="${colspan}" class="no-data">No bird sightings yet. Check back soon!</td></tr>`;
        return;
    }

    sightings.forEach(sighting => {
        const row = document.createElement('tr');

        // Date & Time
        const dateCell = document.createElement('td');
        const fullTimestamp = formatTimestamp(sighting.timestamp, sighting.timezone);
        // Split at the time (last comma) to put time on new line
        const parts = fullTimestamp.split(', ');
        if (parts.length >= 3) {
            // Rejoin date parts, then add time on new line
            const datePart = parts.slice(0, -1).join(', ');
            const timePart = parts[parts.length - 1];
            dateCell.innerHTML = `${datePart}<br>${timePart}`;
        } else {
            dateCell.textContent = fullTimestamp;
        }
        row.appendChild(dateCell);

        // Bird Species (Common + Scientific)
        const speciesCell = document.createElement('td');
        speciesCell.innerHTML = `
          <div>${sighting.common_name}</div>
          <div class="scientific-name">${sighting.scientific_name}</div>
        `;
        row.appendChild(speciesCell);

        // Confidence
        const confidenceCell = document.createElement('td');
        const confidencePercent = (sighting.confidence * 100).toFixed(1);
        confidenceCell.innerHTML = `<span class="confidence ${getConfidenceClass(sighting.confidence)}">${confidencePercent}%</span>`;
        row.appendChild(confidenceCell);

        // Top 3 Predictions
        const predictionsCell = document.createElement('td');
        if (sighting.top_3_predictions && sighting.top_3_predictions.length > 0) {
            const predictionsList = sighting.top_3_predictions
                .map((pred, idx) => `${idx + 1}. ${pred.label} (${(pred.confidence * 100).toFixed(1)}%)`)
                .join('<br>');
            predictionsCell.innerHTML = `<div class="top-predictions">${predictionsList}</div>`;
        } else {
            predictionsCell.textContent = '-';
        }
        row.appendChild(predictionsCell);

        // Admin Actions (only show if logged in as admin)
        if (isAdmin) {
            const actionsCell = document.createElement('td');
            actionsCell.className = 'actions-cell';
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'delete-btn';
            deleteBtn.textContent = 'Delete';
            deleteBtn.onclick = () => deleteSighting(sighting.id, sighting.common_name);
            actionsCell.appendChild(deleteBtn);
            row.appendChild(actionsCell);
        }

        tbody.appendChild(row);
    });
}

function loadSightings() {
    hideError();

    // Query Firestore for all sightings, ordered by timestamp (newest first)
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

            // Update UI
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

// Load sightings when page loads
if (db) {
    loadSightings();
}