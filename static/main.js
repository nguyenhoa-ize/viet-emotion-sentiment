// static/main.js
async function postJSON(url, data) {
  const res = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data)
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function pretty(obj) { return JSON.stringify(obj, null, 2); }

document.addEventListener("DOMContentLoaded", () => {
  const API_BASE = "/api"; // 👈 thêm dòng này

  const text = document.getElementById("text");
  const probs = document.getElementById("probs");
  const btn = document.getElementById("btn");
  const out = document.getElementById("out");

  // ví dụ nhanh - click để đổ lên textarea
  document.querySelectorAll("#examples li").forEach(li => {
    li.addEventListener("click", () => { text.value = li.textContent; });
  });

  btn.addEventListener("click", async () => {
    const payload = { text: text.value.trim(), return_probs: probs.checked };
    if (!payload.text) { out.textContent = "Vui lòng nhập câu."; return; }
    out.textContent = "Đang phân tích...";
    try {
      // 👇 đổi /predict -> /api/predict
      const data = await postJSON(`${API_BASE}/predict`, payload);
      const lines = [
        `Raw: ${payload.text}`,
        `Expanded: ${data.expanded}`,
        `Label: ${data.label}`,
      ];
      if (data.probs) lines.push(`Probs: ${pretty(data.probs)}`);
      out.textContent = lines.join("\n");
    } catch (e) {
      out.textContent = "Lỗi: " + e.message;
    }
  });
});
