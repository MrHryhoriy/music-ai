const form = document.getElementById("form");
const statusEl = document.getElementById("status");
const btn = document.getElementById("btn");
const btnText = document.querySelector(".btnText");

const result = document.getElementById("result");
const headline = document.getElementById("headline");
const rec = document.getElementById("rec");
const genresBox = document.getElementById("genres");
const moodsBox = document.getElementById("moods");

let timerId = null;

function fmtTime(sec) {
  const m = String(Math.floor(sec / 60)).padStart(2, "0");
  const s = String(sec % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function startLoading() {
  btn.disabled = true;
  btn.classList.add("loading");
  btnText.textContent = "Обробляю…";
  result.classList.add("hidden");

  const startedAt = Date.now();
  let dots = 0;
  timerId = setInterval(() => {
    const sec = Math.floor((Date.now() - startedAt) / 1000);
    dots = (dots + 1) % 4;
    statusEl.textContent = `⏳ Виконую аналіз${".".repeat(dots)} (${fmtTime(
      sec
    )})`;
    if (sec === 10)
      statusEl.textContent = `♨️ Перший запуск може бути довшим (${fmtTime(
        sec
      )})`;
  }, 500);
}

function stopLoading(message) {
  if (timerId) clearInterval(timerId);
  timerId = null;
  btn.disabled = false;
  btn.classList.remove("loading");
  btnText.textContent = "Класифікувати";
  statusEl.textContent = message || "";
}

function esc(s) {
  return (s || "").replace(
    /[&<>"']/g,
    (c) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
      }[c])
  );
}

function renderBars(container, items) {
  container.innerHTML = "";
  (items || []).forEach((it) => {
    const pct = Math.max(0, Math.min(100, Number(it.pct || 0)));
    const label = esc(it.label);
    container.insertAdjacentHTML(
      "beforeend",
      `
      <div class="barRow">
        <div class="barTop">
          <div class="barLabel">${label}</div>
          <div class="barPct">${pct.toFixed(1)}%</div>
        </div>
        <div class="barTrack">
          <div class="barFill" style="width:${pct}%"></div>
        </div>
      </div>
    `
    );
  });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  startLoading();

  const fd = new FormData(form);

  try {
    const res = await fetch("/api/predict", { method: "POST", body: fd });
    const text = await res.text();

    // Якщо це JSON — успіх. Якщо ні — це текст помилки.
    let data = null;
    try {
      data = JSON.parse(text);
    } catch (_) {}

    if (!data) {
      stopLoading("❌ " + text);
      return;
    }

    headline.textContent = `${data.top_genre} • ${data.top_mood}`;
    rec.innerHTML = data.recommendation || "✅ Готово";

    renderBars(genresBox, data.genres);
    renderBars(moodsBox, data.moods);

    result.classList.remove("hidden");
    stopLoading("✅ Готово!");
  } catch (err) {
    stopLoading("❌ " + err.message);
  }
});
