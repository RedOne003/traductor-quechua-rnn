// Traductor es->quechua: forward pass manual de una LSTM encoder-decoder,
// ejecutado enteramente en el navegador (sin backend).

let MODEL = null;

async function loadModel() {
  const res = await fetch("model.json");
  MODEL = await res.json();
  document.getElementById("translation").textContent = "Modelo cargado. Escribe una oración.";
}

function normalizeEs(text) {
  text = text.toLowerCase().trim();
  text = text.replace(/([.,!?¿¡;:"'()—–])/g, " $1 ");
  text = text.replace(/[^a-záéíóúñü'.,!?¿¡;:0-9 ]+/gi, " ");
  text = text.replace(/\s+/g, " ").trim();
  return text;
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function tanh(x) { return Math.tanh(x); }

function matVec(mat, vec) {
  // mat: [inDim][outDim] (lista de filas), vec: [inDim] -> devuelve [outDim]
  const outDim = mat[0].length;
  const out = new Array(outDim).fill(0);
  for (let i = 0; i < vec.length; i++) {
    const row = mat[i];
    const vi = vec[i];
    if (vi === 0) continue;
    for (let j = 0; j < outDim; j++) out[j] += vi * row[j];
  }
  return out;
}

function addVec(a, b) { return a.map((v, i) => v + b[i]); }

function lstmStep(x, h, c, kernel, recKernel, bias, units) {
  const gates = addVec(addVec(matVec(kernel, x), matVec(recKernel, h)), bias);
  const i = gates.slice(0, units).map(sigmoid);
  const f = gates.slice(units, 2 * units).map(sigmoid);
  const cHat = gates.slice(2 * units, 3 * units).map(tanh);
  const o = gates.slice(3 * units, 4 * units).map(sigmoid);
  const newC = f.map((fv, k) => fv * c[k] + i[k] * cHat[k]);
  const newH = newC.map((cv, k) => o[k] * tanh(cv));
  return [newH, newC];
}

function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

function encode(tokenIds) {
  const units = MODEL.units;
  const [embMatrix] = MODEL.encoder_embedding;
  const [kernel, recKernel, bias] = MODEL.encoder_lstm;
  let h = new Array(units).fill(0);
  let c = new Array(units).fill(0);
  for (const id of tokenIds) {
    const x = embMatrix[id];
    [h, c] = lstmStep(x, h, c, kernel, recKernel, bias, units);
  }
  return [h, c];
}

function decode(h, c) {
  const units = MODEL.units;
  const [embMatrix] = MODEL.decoder_embedding;
  const [kernel, recKernel, bias] = MODEL.decoder_lstm;
  const [denseKernel, denseBias] = MODEL.decoder_output;
  const words = [];
  let token = MODEL.sos_id;
  for (let step = 0; step < MODEL.max_len; step++) {
    const x = embMatrix[token];
    [h, c] = lstmStep(x, h, c, kernel, recKernel, bias, units);
    const logits = addVec(matVec(denseKernel, h), denseBias);
    const nextId = argmax(logits);
    if (nextId === MODEL.eos_id || nextId === 0) break;
    words.push(MODEL.index_word_qu[String(nextId)] || "<unk>");
    token = nextId;
  }
  return words.join(" ");
}

function translate(sentence) {
  const norm = normalizeEs(sentence);
  const words = norm.split(" ").filter(Boolean);
  const unkId = MODEL.word_index_es["<unk>"];
  const sosId = MODEL.word_index_es["<sos>"];
  const eosId = MODEL.word_index_es["<eos>"];

  const known = words.map(w => [w, Object.prototype.hasOwnProperty.call(MODEL.word_index_es, w)]);
  const ids = [sosId, ...words.map(w => MODEL.word_index_es[w] ?? unkId), eosId];

  const [h, c] = encode(ids);
  const translation = decode(h, c);
  return { known, translation };
}

function renderKnown(known) {
  const el = document.getElementById("words");
  if (known.length === 0) { el.innerHTML = "—"; return; }
  el.innerHTML = known.map(([w, ok]) =>
    `<span class="${ok ? "known" : "unknown"}">${w}${ok ? "" : " (?)"}</span>`).join("");
}

function run() {
  const text = document.getElementById("input").value;
  if (!MODEL) { document.getElementById("translation").textContent = "Cargando modelo..."; return; }
  if (!text.trim()) return;
  const { known, translation } = translate(text);
  renderKnown(known);
  document.getElementById("translation").textContent = translation || "(no se pudo traducir, intenta con otra oración corta)";
}

const EJEMPLOS = ["Dios te ama", "Vamos a la escuela", "Mi casa es grande", "Buenos días", "Gracias por tu ayuda"];

window.addEventListener("DOMContentLoaded", () => {
  loadModel();
  document.getElementById("btn").addEventListener("click", run);
  document.getElementById("input").addEventListener("keydown", e => { if (e.key === "Enter") run(); });
  const exBox = document.getElementById("examples");
  EJEMPLOS.forEach(ex => {
    const b = document.createElement("button");
    b.textContent = ex;
    b.onclick = () => { document.getElementById("input").value = ex; run(); };
    exBox.appendChild(b);
  });
});
