import sys
import subprocess
import importlib
import tkinter as tk
from tkinter import messagebox, simpledialog

# 📦 Required Libraries
required_libs = ["sklearn"]

# ⚙️ Autoload Missing Libraries
def autoload_libs():
    for lib in required_libs:
        try:
            importlib.import_module(lib)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            except Exception as e:
                messagebox.showerror("Library Error", f"Failed to install {lib}: {e}")

autoload_libs()

# 🔍 Import AI Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 🎨 GUI Setup
root = tk.Tk()
root.title("🧙‍♂️ MagicBox Hybrid Assistant")
root.geometry("600x580")
root.configure(bg="#1e1e2f")

font_style = ("Helvetica", 16, "bold")
btn_style = {
    "font": font_style,
    "bg": "#5c5c8a",
    "fg": "white",
    "activebackground": "#7070a0",
    "width": 25,
    "height": 2
}

# 🧠 Tab Clustering Logic
sample_tabs = [
    "How to bake sourdough",
    "Bread recipes and techniques",
    "Python clustering tutorials",
    "Machine learning for beginners",
    "Top gardening tips"
]

def cluster_tab_titles(titles, num_clusters=2):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(titles)
    model = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = model.fit_predict(X)
    clustered = {}
    for i, label in enumerate(clusters):
        clustered.setdefault(label, []).append(titles[i])
    return clustered

# 💬 Conversational Brain
def ai_response(user_input):
    user_input = user_input.lower()
    if "hello" in user_input:
        return "Hello, wise one! Ready to sort some tabs?"
    elif "python" in user_input:
        return "Python is the wand. What spell would you like?"
    elif "cluster" in user_input:
        return "Tab clustering is ready. Shall we begin?"
    else:
        return "Hmm... I sense the answer lies deeper in the code scrolls."

# 🖱️ Button Functions
def launch_feature():
    messagebox.showinfo("Launch", "✨ AI Assistant activated!")

def cluster_tabs():
    result = cluster_tab_titles(sample_tabs, num_clusters=2)
    msg = ""
    for group, titles in result.items():
        msg += f"\nGroup {group + 1}:\n" + "\n".join(f" - {t}" for t in titles)
    messagebox.showinfo("Tab Clustering", msg.strip())

def talk_to_ai():
    query = simpledialog.askstring("Conversational Assistant", "Speak your mind:")
    if query:
        reply = ai_response(query)
        messagebox.showinfo("Response", reply)

# 🧓 Senior Mode Toggle
def toggle_senior_mode():
    root.configure(bg="black")
    for widget in root.winfo_children():
        if isinstance(widget, tk.Button):
            widget.config(font=("Helvetica", 18, "bold"))

# 🎯 GUI Layout
tk.Label(root, text="MagicBox Hybrid Assistant", font=("Helvetica", 20, "bold"), fg="white", bg="#1e1e2f").pack(pady=20)

tk.Button(root, text="🚀 Launch AI Assistant", command=launch_feature, **btn_style).pack(pady=10)
tk.Button(root, text="🗂️ Tab Clustering Wizard", command=cluster_tabs, **btn_style).pack(pady=10)
tk.Button(root, text="💬 Conversational Assistant", command=talk_to_ai, **btn_style).pack(pady=10)
tk.Button(root, text="👴 Senior Mode Toggle", command=toggle_senior_mode, **btn_style).pack(pady=10)

# 🔚 Start GUI Loop
root.mainloop()

