const socket = io();

socket.on("oracle_update", data => {
  document.getElementById("oracle-utterance").innerText = data.utterance;
  logGlyph(data.utterance);
});

function emitAction(mode) {
  socket.emit("gui_action", { mode: mode });
}

function logGlyph(msg) {
  const bank = document.getElementById("glyph-bank");
  const card = document.createElement("div");
  card.className = "glyph-card";
  card.innerText = msg;
  bank.prepend(card);
}

