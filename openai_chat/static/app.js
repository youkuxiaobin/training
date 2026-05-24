const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chat-form");
const promptEl = document.querySelector("#prompt");
const modelEl = document.querySelector("#model");
const maxTokensEl = document.querySelector("#max-output-tokens");
const statusEl = document.querySelector("#status");
const sendEl = document.querySelector("#send");
const clearEl = document.querySelector("#clear");

const messages = [];

function appendMessage(role, content) {
  const item = document.createElement("div");
  item.className = `message ${role}`;
  item.textContent = content;
  messagesEl.appendChild(item);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setBusy(isBusy) {
  sendEl.disabled = isBusy;
  promptEl.disabled = isBusy;
  statusEl.textContent = isBusy ? "Thinking" : "Ready";
}

function appendError(content) {
  const item = document.createElement("div");
  item.className = "message error";
  item.textContent = content;
  messagesEl.appendChild(item);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMessage(content) {
  messages.push({ role: "user", content });
  appendMessage("user", content);
  setBusy(true);

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: modelEl.value.trim(),
      messages,
      max_output_tokens: Number(maxTokensEl.value || 512),
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Request failed");
  }

  messages.push({ role: "assistant", content: payload.reply });
  appendMessage("assistant", payload.reply);
  statusEl.textContent = payload.model || "Ready";
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = promptEl.value.trim();
  if (!content) {
    return;
  }
  promptEl.value = "";
  try {
    await sendMessage(content);
  } catch (error) {
    appendError(error.message);
    statusEl.textContent = "Error";
  } finally {
    setBusy(false);
    promptEl.focus();
  }
});

clearEl.addEventListener("click", () => {
  messages.length = 0;
  messagesEl.replaceChildren();
  statusEl.textContent = "Ready";
  promptEl.focus();
});
