// ============================================
// IMPORTS
// ============================================
import "https://www.gstatic.com/firebasejs/10.7.1/firebase-app-compat.js";
import "https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore-compat.js";


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
try {
    firebase.initializeApp(firebaseConfig);
    db = firebase.firestore();
    console.log("Firebase initialized successfully");
} catch (error) {
    console.error("Error initializing Firebase:", error);
    showError("Failed to connect to Firebase. Please check your configuration.");
}

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
        tbody.innerHTML = '<tr><td colspan="4" class="no-data">No bird sightings yet. Check back soon!</td></tr>';
        return;
    }

    sightings.forEach(sighting => {
        const row = document.createElement('tr');

        // Date & Time
        const dateCell = document.createElement('td');
        dateCell.textContent = formatTimestamp(sighting.timestamp, sighting.timezone);
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