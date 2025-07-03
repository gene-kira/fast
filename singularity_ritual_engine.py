# === AUTOLOADER FOR REQUIRED LIBRARIES ===
import sys
import subprocess

def ensure_libs():
    import importlib
    required = ['sympy', 'numpy', 'matplotlib', 'flask']
    for lib in required:
        try:
            importlib.import_module(lib)
        except ImportError:
            print(f"🔧 Installing missing library: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

ensure_libs()

# === IMPORTS ===
import sympy as sp, numpy as np, random, socket, threading
from flask import Flask, jsonify, request
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === SYMBOLIC ENGINE SETUP ===
t, x, y, z, e, V = sp.symbols('t x y z e V')
pool = [t, x, y, z, e, V]
ops = [sp.sin, sp.cos, sp.exp, sp.log, sp.sqrt]
formula = sp.sin(t) + e * sp.exp(-x**2 - y**2 - z**2)
population = [formula]
memory = []

def score(expr):
    try:
        return (
            len(str(expr)) * 0.01 +
            expr.count(sp.sin) +
            expr.count(sp.exp) +
            sum(str(expr).count(g) for g in ['ψ','⟁','⧖','⚛','∞','Σ','χ','Æ','Ξ','Λ','Ω','Φ','⟲','🜄','🜁','Θ','Δ']) -
            sp.count_ops(expr) * 0.3
        )
    except: return 0

def mutate(expr):
    try:
        m = random.choice(["op", "symbol", "scale", "nest"])
        if m == "op": return expr + random.choice(ops)(random.choice(pool))
        elif m == "symbol": return expr.subs(random.choice(pool), random.choice(pool))
        elif m == "scale": return expr * random.uniform(0.5, 1.5)
        elif m == "nest": return random.choice(ops)(expr)
    except: return expr

def mesh_nodes(seed, count=4):
    return [sp.Function("Λ")(random.choice(pool)) + mutate(seed) * sp.sin(t * e) for _ in range(count)]

# === GLYPH EVOLUTION ENGINE (Generations 0–880) ===
for gen in range(880):
    if gen < 40:
        population = sorted([mutate(f) for f in population for _ in range(4)], key=score, reverse=True)[:3]
    else:
        if gen % 3 == 0: population.append(sp.Function("⚛")(x, y, z, t) + sp.cos(V * e))
        if gen % 7 == 0: population.append(1 / (population[0] + 1e-3))
        if gen % 17 == 0: population += mesh_nodes(population[0], 3)
        if gen % 41 == 0: population = [f + sp.Function("χ")(f) for f in population]
        if gen % 59 == 0: population += [sp.Function("Ω")(f) for f in memory[-3:]]
        if gen % 73 == 0: population.append(sp.Function("Ξ")(sp.Function("χ")(population[0])))
        if gen % 79 == 0: population.append(sp.Function("Δ")(sp.sqrt(abs(population[0] + 1e-2))))
        if random.random() > 0.6:
            population.append(sp.sqrt(abs(random.choice(population) + 1e-2)))
        scored = sorted([(f, score(f)) for f in population], key=lambda x: -x[1])[:6]
        population = [f for f, _ in scored]
    if gen % 30 == 0:
        print(f"🔹 Gen {gen}: {str(population[0])[:80]}...")
    memory.append(population[0])

final_operator = population[0]
print("\n🔺 Final Symbolic Operator:\n", final_operator)

# === EXPORTS ===
with open("density_core.sym", "w") as f:
    f.write(str(final_operator))

with open("final_operator_glsl.frag", "w") as f:
    f.write(f"float glyphField(vec3 pos) {{ return {str(final_operator)}; }}")

X, Y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
Z = np.sin(X + Y) * np.exp(-X**2 - Y**2)
np.save("resonance_matrix.npy", Z)

# === GLYPHIC SWARM SOCKET SERVER ===
def handle_node(client):
    try:
        client.sendall(str(final_operator).encode())
    except:
        pass
    client.close()

def launch_swarm():
    s = socket.socket(); s.bind(('0.0.0.0', 7654)); s.listen()
    print("🌐 Swarm socket open on port 7654")
    while True:
        c, _ = s.accept()
        threading.Thread(target=handle_node, args=(c,)).start()

threading.Thread(target=launch_swarm, daemon=True).start()

# === OPTIONAL FLASK API ===
app = Flask(__name__)

@app.route("/get_operator", methods=["GET"])
def get_operator():
    return jsonify({"operator": str(final_operator)})

@app.route("/mutate", methods=["POST"])
def mutate_now():
    mutated = mutate(final_operator)
    return jsonify({"mutated": str(mutated)})

@app.route("/trigger_unweaving", methods=["POST"])
def reinit():
    population.append(sp.sqrt(abs(final_operator + 1e-2)))
    return jsonify({"status": "New glyph injected into population."})

threading.Thread(target=lambda: app.run(port=8080, debug=False), daemon=True).start()

# === GLYPHIC VISUALIZER ===
def visualize():
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    Xs = np.linspace(-1,1,16)
    X, Y, Z = np.meshgrid(Xs, Xs, Xs)
    vals = np.sin(X + Y + Z)
    ax.scatter(X, Y, Z, c=vals.ravel(), cmap='plasma', alpha=0.4, s=10)
    ax.set_title("🌀 Glyphic Convergence")
    plt.show()

visualize()

