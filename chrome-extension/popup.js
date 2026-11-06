/**
 * Popup Script 
 */
const API_URL = "http://localhost:8000/predict_batch";
// const API_URL   = "https://organic-space-broccoli-wrgg99ww6gqqh5659-7860.app.github.dev/predict_batch";
const SENTIMENT = { "Positif": 1, "Neutre": 0, "Négatif": -1 };
const COLORS    = { "Positif": "#22c55e", "Neutre": "#f59e0b", "Négatif": "#ef4444" };

let allResults   = [];
let activeFilter = "all";

// ── Utilitaires DOM ──────────────────────────────────────────────────────────

const $  = id => document.getElementById(id);
const show = id => $(id).classList.remove("hidden");
const hide = id => $(id).classList.add("hidden");

// ── Dark mode ────────────────────────────────────────────────────────────────

$("themeToggle").addEventListener("click", () => {
  document.body.classList.toggle("dark");
  $("themeToggle").textContent = document.body.classList.contains("dark") ? "☀️" : "🌙";
  chrome.storage.local.set({ dark: document.body.classList.contains("dark") });
});

chrome.storage.local.get("dark", ({ dark }) => {
  if (dark) {
    document.body.classList.add("dark");
    $("themeToggle").textContent = "☀️";
  }
});

// ── Analyse principale ───────────────────────────────────────────────────────

$("analyzeBtn").addEventListener("click", async () => {
  hide("error"); hide("results");
  show("loading");
  $("analyzeBtn").disabled = true;

  try {
    // 1. Récupérer les commentaires via content.js
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab.url.includes("youtube.com/watch")) {
      throw new Error("Ouvre une vidéo YouTube pour analyser ses commentaires.");
    }

    const { comments } = await chrome.tabs.sendMessage(tab.id, { action: "getComments" });

    if (!comments || comments.length === 0) {
      throw new Error("Aucun commentaire trouvé. Fais défiler la page pour en charger.");
    }

    // 2. Appel API
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments }),
    });

    if (!res.ok) throw new Error(`Erreur API : ${res.status}`);
    const data = await res.json();

    allResults = data.results;
    renderStats(data.stats, data.total);
    renderList(allResults);
    hide("loading"); show("results");

  } catch (err) {
    hide("loading");
    $("errorMsg").textContent = err.message;
    show("error");
  } finally {
    $("analyzeBtn").disabled = false;
  }
});

// ── Stats + graphique ────────────────────────────────────────────────────────

function renderStats(stats, total) {
  $("totalCount").textContent = total;

  ["Positif", "Neutre", "Négatif"].forEach(label => {
    const s = stats[label] || { count: 0, pct: 0 };
    $(`count${label}`).textContent = s.count;
    $(`pct${label}`).textContent   = `${s.pct}%`;
    const bar = $(`bar${label}`);
    bar.style.width = `${s.pct}%`;
    bar.style.background = COLORS[label];
  });
}

// ── Liste des commentaires ───────────────────────────────────────────────────

function renderList(results) {
  const container = $("commentList");
  container.innerHTML = "";

  const filtered = activeFilter === "all"
    ? results
    : results.filter(r => r.sentiment === activeFilter);

  if (filtered.length === 0) {
    container.innerHTML = `<p class="empty">Aucun commentaire pour ce filtre.</p>`;
    return;
  }

  filtered.forEach(({ comment, sentiment, confidence }) => {
    const div = document.createElement("div");
    div.className = "comment-item";
    div.innerHTML = `
      <div class="comment-header">
        <span class="badge" style="background:${COLORS[sentiment]}">${sentiment}</span>
        <span class="confidence">${(confidence * 100).toFixed(1)}%</span>
      </div>
      <p class="comment-text">${escapeHtml(comment)}</p>
    `;
    container.appendChild(div);
  });
}

// ── Filtres ──────────────────────────────────────────────────────────────────

document.querySelectorAll(".filter-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    activeFilter = btn.dataset.filter;
    renderList(allResults);
  });
});

// ── Copier les résultats ─────────────────────────────────────────────────────

$("copyBtn").addEventListener("click", () => {
  const lines = allResults.map(r =>
    `[${r.sentiment}] (${(r.confidence * 100).toFixed(1)}%) ${r.comment}`
  );
  navigator.clipboard.writeText(lines.join("\n")).then(() => {
    $("copyBtn").textContent = "Copié !";
    setTimeout(() => $("copyBtn").textContent = "Copier", 1500);
  });
});

// ── Helpers ──────────────────────────────────────────────────────────────────

function escapeHtml(text) {
  return text.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}