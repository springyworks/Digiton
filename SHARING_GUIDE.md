# How to Share Interactive 3D Digiton Visualizations

## ✅ Best Options (Ranked by Trust & Ease)

### 1. **GitHub Pages** (RECOMMENDED - Most Trusted)
- **Trust Level:** ⭐⭐⭐⭐⭐ (People trust `github.io` domains)
- **Cost:** Free
- **Setup Time:** 2 minutes

**Steps:**
```bash
# In your Digiton repo
git checkout -b gh-pages
cp data/*.html docs/
git add docs/
git commit -m "Add interactive visualizations"
git push origin gh-pages
```

Then enable GitHub Pages in repo settings → Pages → Source: `gh-pages` branch → `/docs` folder.

**Share URL:** `https://yourusername.github.io/Digiton/05_3d_corkscrew.html`

**Why it works:**
- `.github.io` domain is trusted
- People can see it's from your official GitHub
- No suspicious file downloads
- Works on mobile too

---

### 2. **GitHub Gist** (Quick & Simple)
- **Trust Level:** ⭐⭐⭐⭐
- **Cost:** Free
- **Setup Time:** 30 seconds

**Steps:**
1. Go to https://gist.github.com
2. Paste the HTML content
3. Name it `digiton_3d_viz.html`
4. Create Public Gist
5. Click "View Raw" → Copy that URL

**Share URL:** `https://gist.githubusercontent.com/yourusername/abc123.../raw/.../digiton_3d_viz.html`

**Add `bl.ocks.org` viewer:**
- Share: `https://bl.ocks.org/yourusername/gist-id`
- This renders the HTML in an iframe (trusted by data viz community)

---

### 3. **Observable** (Data Viz Platform)
- **Trust Level:** ⭐⭐⭐⭐⭐
- **Cost:** Free
- **Setup Time:** 5 minutes
- **URL:** https://observablehq.com

**Why:** Built for interactive data visualizations. Plotly works natively. People in signal processing/radio community use it.

**Convert HTML to Observable:**
```javascript
// In an Observable notebook:
import {Plot} from "@observablehq/plot"
// Or embed Plotly directly
html`${yourPlotlyHTML}`
```

---

### 4. **Google Drive** (Works, but Less Trusted)
- **Trust Level:** ⭐⭐⭐ (People hesitate to open HTML from Drive)
- **Cost:** Free
- **Limitation:** Drive shows a preview page, not the actual interactive HTML

**How it works:**
1. Upload `05_3d_corkscrew.html` to Google Drive
2. Right-click → Share → Get Link → "Anyone with link can view"
3. Get the file ID from URL: `https://drive.google.com/file/d/FILE_ID/view`
4. Share as: `https://drive.google.com/uc?export=download&id=FILE_ID`

**Problem:** People have to download it first. They see a warning. They won't trust it unless they know you.

---

### 5. **Netlify Drop** (Zero Config Hosting)
- **Trust Level:** ⭐⭐⭐⭐
- **Cost:** Free
- **Setup Time:** 10 seconds
- **URL:** https://app.netlify.com/drop

**Steps:**
1. Drag the HTML file into Netlify Drop
2. Get a URL like `https://random-name-123.netlify.app/05_3d_corkscrew.html`

**Pros:**
- Instant deployment
- HTTPS by default
- Trusted domain

---

## 🛡️ Making People Trust Your HTML

### Add a README on the same platform:
```markdown
# Digiton 3D Visualization

This is an **interactive 3D plot** of radio signal analysis.

**Safe to open:** 
- Pure HTML/JavaScript (no server-side code)
- Uses Plotly.js (open-source library from plotly.com)
- No tracking, no ads, no data collection
- Source code: [link to your repo]

**What you'll see:**
A 3D "corkscrew" visualization of Gaussian pulses spinning in I/Q space.
```

### Add metadata to HTML:
Add this at the top of your HTML file (after `<head>`):
```html
<meta name="description" content="Interactive 3D visualization of Digiton radio signals">
<meta name="author" content="Your Name">
<meta property="og:title" content="Digiton 3D Corkscrews">
<meta property="og:description" content="SSB I/Q signal analysis">
```

---

## 🎯 Recommendation for Radio/Ham Community

**Use GitHub Pages + README**

1. Create `docs/` folder in your repo
2. Put all HTML files there
3. Add `docs/README.md` explaining what each file is
4. Enable GitHub Pages
5. Share: `https://yourusername.github.io/Digiton/`

**Social Media Post Template:**
```
🌀 New: Interactive 3D visualization of my Digiton protocol!

See the "spin" of Gaussian pulses in I/Q space (the SSB trick).
Fully interactive - rotate, zoom, fly around.

🔗 https://yourusername.github.io/Digiton/05_3d_corkscrew.html
📂 Source: https://github.com/yourusername/Digiton

Built with Python + Plotly. No downloads, opens in browser.
Safe to click - it's just a static HTML visualization.
```

---

## ⚡ Quick Command (GitHub Pages Setup)

```bash
cd /home/rustuser/projects/pyth/Digiton
mkdir -p docs
cp data/*.html docs/
cp data/*.png docs/
cat > docs/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head><title>Digiton Visualizations</title></head>
<body>
<h1>Digiton: Spin Modem Visualizations</h1>
<ul>
  <li><a href="05_3d_corkscrew.html">3D Interactive Corkscrews</a></li>
  <li><a href="01_spin_digiton_modem.png">Spin Modem</a></li>
  <li><a href="02_digiton_chat_spin.png">Chat Simulation</a></li>
</ul>
</body>
</html>
EOF

git add docs/
git commit -m "Add interactive visualizations"
git push
```

Then enable Pages in GitHub repo settings.
