# === ritual_4d_plot.py ===
# Visualize 4D symbolic cognition using glyphic swarm agents

# --- AUTOLOADER ---
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    import random
except ImportError as e:
    print(f"Missing library: {e.name}")
    exit()

# --- SWARM NODE GENERATOR ---
def generate_nodes(n=50):
    nodes = []
    for _ in range(n):
        node = {
            "x": random.uniform(-1, 1),
            "y": random.uniform(-1, 1),
            "z": random.uniform(-1, 1),
            "entropy": random.uniform(0, 1),  # 4th dimension
            "glyph": random.choice(["⚛", "⟁", "✴", "☢", "⊕", "⧖", "◉"]),
        }
        nodes.append(node)
    return nodes

# --- PLOTTING FUNCTION ---
def plot_nodes_4d_swarm(nodes, frames=80):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter([], [], [], c=[], cmap='plasma', s=70)
    texts = []

    def init():
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_title("🔮 4D Ritual Glyph Swarm\n(X,Y,Z) + Entropy → Color")
        return sc,

    def update(frame):
        xs, ys, zs, cs = [], [], [], []
        ax.collections.clear()
        for txt in texts:
            txt.remove()
        texts.clear()

        for node in nodes:
            # Animate motion + 4th dimension
            node["x"] += 0.01 * np.sin(frame / 10 + random.random())
            node["y"] += 0.01 * np.cos(frame / 10 + random.random())
            node["z"] += 0.01 * np.sin(frame / 15)
            node["entropy"] = abs(np.sin(frame / 10 + random.random()))

            xs.append(node["x"])
            ys.append(node["y"])
            zs.append(node["z"])
            cs.append(node["entropy"])

        sc = ax.scatter(xs, ys, zs, c=cs, cmap='plasma', s=70)

        for i, node in enumerate(nodes[:12]):  # Just a few glyphs
            txt = ax.text(xs[i], ys[i], zs[i], node["glyph"], fontsize=14)
            texts.append(txt)
        return sc,

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, interval=120, blit=False)
    plt.show()

# === BOOTSTRAP ===
if __name__ == "__main__":
    swarm = generate_nodes()
    plot_nodes_4d_swarm(swarm)

