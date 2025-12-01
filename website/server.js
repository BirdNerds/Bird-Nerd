const express = require("express");
const fs = require("fs");
const path = require("path");

const app = express();

// Helper: convert YYYYMMDD + HHMMSS → readable date
function formatDate(dateStr, timeStr) {
    const year = dateStr.substring(0, 4);
    const month = dateStr.substring(4, 6);
    const day = dateStr.substring(6, 8);

    const hour = timeStr.substring(0, 2);
    const minute = timeStr.substring(2, 4);

    const date = new Date(`${year}-${month}-${day}T${hour}:${minute}:00`);

    return date.toLocaleString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
    });
}


// List all image entries with readable display names
app.get("/", (req, res) => {
    const files = fs.readdirSync(CLASSIFIED_DIR)
        .filter(f => f.endsWith(".jpg"))
        .map(f => f.replace(".jpg", ""));

    let html = "<h1>Classified Images</h1><ul>";

    for (const base of files) {

        // Example filename:
        // bird-Agelaius phoeniceus (Red-winged Blackbird)_20251027_174326_192.00

        const parts = base.split("_");
        const speciesFull = parts[0].replace("bird-", ""); 
        const dateStr = parts[1];
        const timeStr = parts[2];

        const formatted = formatDate(dateStr, timeStr);

        html += `
            <li>
               <a href="/view/${base}">
                    ${speciesFull} — ${formatted}
               </a>
            </li>
        `;
    }

    html += "</ul>";

    res.send(html);
});


// Path to your classified folder
const CLASSIFIED_DIR = path.join(__dirname, "../motion_camera/classified");

// Serve images directly from the classified folder
app.use("/classified", express.static(CLASSIFIED_DIR));


// Route: show image and its text file
app.get("/view/:basename", (req, res) => {
    const basename = req.params.basename; // everything before extension

    const imgPath = path.join(CLASSIFIED_DIR, basename + ".jpg");
    const txtPath = path.join(CLASSIFIED_DIR, basename + ".txt");

    // Check if image exists
    if (!fs.existsSync(imgPath)) {
        return res.status(404).send("Image not found");
    }

    // Read text file (if it exists)
    let txtContent = "No info available.";
    if (fs.existsSync(txtPath)) {
        txtContent = fs.readFileSync(txtPath, "utf8");
    }

    // Send a simple HTML page
    res.send(`
        <html>
        <head>
            <title>${basename}</title>
            <style>
                body { font-family: Arial; padding: 20px; }
                img { max-width: 600px; border-radius: 8px; }
                pre { background: #f3f3f3; padding: 10px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h2>${basename}</h2>
            <img src="/classified/${basename}.jpg" /><br><br>

            <h3>Detection Info</h3>
            <pre>${txtContent}</pre>
        </body>
        </html>
    `);
});


// List all image entries
app.get("/", (req, res) => {
    const files = fs.readdirSync(CLASSIFIED_DIR)
        .filter(f => f.endsWith(".jpg"))
        .map(f => f.replace(".jpg", ""));

    let html = "<h1>Classified Images</h1><ul>";
    for (const base of files) {
        html += `<li><a href="/view/${base}">${base}</a></li>`;
    }
    html += "</ul>";

    res.send(html);
});


app.listen(3000, () => {
    console.log("Server running on http://localhost:3000");
});
