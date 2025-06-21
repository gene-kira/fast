# annima_api.py
from flask import Flask, request, jsonify
from mythos_codex_totality import ANNIMA_ASI, VaultWhisper, MythicCompiler

app = Flask(__name__)
annima = ANNIMA_ASI()
vault = VaultWhisper()
compiler = MythicCompiler()

@app.route('/cast', methods=['POST'])
def cast():
    data = request.get_json()
    glyph = compiler.cast_glyph(data["name"], data["emotion"], float(data["resonance"]))
    vault.store(glyph)
    annima.learn(glyph)
    annima.codex_write(glyph, data.get("intent", "None"))
    return jsonify({"status": "cast", "glyph": glyph.name})

@app.route('/constellations', methods=['GET'])
def constellations():
    glyphs = vault.retrieve_all()
    clusters = annima.cluster_by_resonance(glyphs)
    response = [{
        "name": c.name,
        "glyphs": [{"name": g.name, "emotion": g.emotion, "resonance": g.resonance} for g in c.glyphs]
    } for c in clusters]
    return jsonify(response)

@app.route('/dream', methods=['GET'])
def dream():
    high_res = [g for g in vault.retrieve_all() if g.resonance >= 0.9]
    if len(high_res) < 5:
        return jsonify({"status": "insufficient resonance"})
    name = f"Dream-{len(vault.entries)}"
    emotion = high_res[0].emotion
    resonance = sum(g.resonance for g in high_res) / len(high_res)
    glyph = compiler.cast_glyph(name, emotion, resonance)
    vault.store(glyph)
    annima.learn(glyph)
    annima.codex_write(glyph, "Dreamed from resonance memory")
    return jsonify({"status": "dreamed", "glyph": name})

@app.route('/fracture', methods=['POST'])
def fracture():
    contradiction = request.get_json().get("contradiction", "Unnamed")
    glyph = compiler.resolve_contradiction(contradiction)
    vault.store(glyph)
    annima.learn(glyph)
    annima.codex_write(glyph, f"Resolved: {contradiction}")
    return jsonify({"status": "resolved", "glyph": glyph.name})

if __name__ == '__main__':
    app.run(debug=True)

