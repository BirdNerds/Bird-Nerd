const express = require("express");
const fs = require("fs");
const path = require("path");

const app = express();

// ----------- PATHS -----------------
const PUBLIC_DIR = path.join(__dirname, "public");
const IMAGE_DIR = path.join(__dirname, "../motion_camera/images");
const LOG_FILE = path.join(__dirname, "../motion_camera/sightings.log");

// ----------- STATIC FILES ----------
app.use(express.static(PUBLIC_DIR));          // serves index.html, script.js, styles.css
app.use("/images", express.static(IMAGE_DIR)); // serves the bird photos directly

// ----------- UTILITIES --------------

/**
 * Convert timestamp "2025-11-25 10:40:30" → "2025_11_25_10_40_30"
 * to match image file names.
 */
function timestampToPrefix(ts) {
    let [date, time] = ts.split(" ");
    return date.replace(/-/g, "_") + "_" + time.replace(/:/g, "_");
}

/**
 * Parse birds.log blocks into objects like:
 * {
 *   datetime: "...",
 *   species: "...",
 *   confidence: "...",
 *   notes: "Top 3 predictions..."
 * }
 */
function parseLogFile() {
    if (!fs.existsSync(LOG_FILE)) return [];

    const raw = fs.readFileSync(LOG_FILE, "utf8");

    const blocks = raw
        .split("**********************************************************************")
        .map(b => b.trim())
        .filter(Boolean);

    const parsed = [];

    for (const block of blocks) {
        const lines = block.split("\n").map(l => l.trim()).filter(Boolean);
        if (lines.length < 3) continue;

        const datetime = lines[0];
        const species = lines[1];
        const confidence = lines[2];
        const notes = lines.slice(3).join("\n");

        parsed.push({ datetime, species, confidence, notes });
    }
    return parsed;
}

// ----------- API: RETURN JSON FOR GALLERY -------------

app.get("/api/photos", (req, res) => {
    const entries = parseLogFile();

    const images = fs.readdirSync(IMAGE_DIR)
        .filter(f => f.endsWith(".jpg"));

    const results = [];

    for (const entry of entries) {
        const prefix = timestampToPrefix(entry.datetime);

        const match = images.find(img => img.startsWith(prefix));
        if (!match) continue;

        results.push({
            url: "/images/" + match,
            filename: match,
            species: entry.species,
            date: entry.datetime.split(" ")[0],
            time: entry.datetime.split(" ")[1],
            notes: entry.notes
        });
    }

    // newest → oldest
    results.sort((a, b) =>
        (a.date + a.time < b.date + b.time ? 1 : -1)
    );

    res.json(results);
});

// ----------- SIMPLE VIEW PAGE -----------

app.get("/view/:basename", (req, res) => {
    const basename = req.params.basename;

    const jpg = path.join(IMAGE_DIR, basename + ".jpg");

    if (!fs.existsSync(jpg)) {
        return res.status(404).send("Image not found");
    }

    res.send(`
        <html>
        <head>
            <title>${basename}</title>
            <style>
                body { font-family: Arial; padding: 20px; }
                img { max-width: 600px; border-radius: 8px; }
            </style>
        </head>
        <body>
            <h2>${basename}</h2>
            <img src="/images/${basename}.jpg" />
            <p><a href="/">Back to gallery</a></p>
        </body>
        </html>
    `);
});

// ----------- START SERVER --------------

app.listen(3000, () => {
    console.log("Bird gallery running at http://localhost:3000");
});