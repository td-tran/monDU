const apiUrl = "http://127.0.0.1:8000/search";
const form = document.getElementById("searchForm");
const queryInput = document.getElementById("query");
const rerankerSelect = document.getElementById("reranker");
const resultsEl = document.getElementById("results");
const statusEl = document.getElementById("status");

function renderResults(data) {
  if (!data.results || data.results.length === 0) {
    resultsEl.innerHTML = "<p>Aucun résultat trouvé.</p>";
    return;
  }
  const html = data.results
    .map(
      (item) => `
      <article class="result-card">
        <h2>${item.title}</h2>
        <p><strong>Université :</strong> ${item.university}</p>
        <p><strong>Lien :</strong> <a href="${item.site}" target="_blank" rel="noreferrer">Voir la formation</a></p>
        <p class="score">Score : ${item.combined_score.toFixed(3)}</p>
      </article>`
    )
    .join("\n");
  resultsEl.innerHTML = html;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  statusEl.textContent = "Recherche en cours...";
  resultsEl.innerHTML = "";

  try {
    const reranker = rerankerSelect.value;
    const response = await fetch(`${apiUrl}?query=${encodeURIComponent(query)}&k=15&reranker=${encodeURIComponent(reranker)}`);
    if (!response.ok) {
      throw new Error(`Erreur serveur ${response.status}`);
    }
    const data = await response.json();
    statusEl.textContent = `Résultats pour « ${data.query} » (${data.count}) — reranker : ${data.reranker}`;
    renderResults(data);
  } catch (error) {
    statusEl.textContent = `Erreur : ${error.message}`;
  }
});
