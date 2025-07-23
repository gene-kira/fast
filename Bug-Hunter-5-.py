# 🔮 MagicBox ASI Agent - Old Guy Friendly Edition
import os, re, json, time, sys
try:
    import pygame
    import tkinter as tk
    from tkinter import messagebox
    from tkinter import ttk
    from tkinter.ttk import Combobox
    from tkinter import Frame, Canvas, Text, Button
    from ttkthemes import ThemedTk
except ImportError as e:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame", "ttkthemes"])
    from ttkthemes import ThemedTk
    from tkinter import messagebox
    from tkinter import ttk
    from tkinter.ttk import Combobox
    from tkinter import Frame, Canvas, Text, Button
    import pygame
    import tkinter as tk

pygame.init()
pygame.mixer.init()

def speak(message):
    try:
        sound = pygame.mixer.Sound("beep.wav")
        sound.play()
    except:
        pass
    print(f"[MagicBox]: {message}")

def play_persona_sound(action):
    speak(f"Persona action triggered: {action}")

PERSONAS = ["Oracle", "Synth", "Archivist", "Rebel"]
current_persona = PERSONAS[0]
current_proposal_id = "001"

app = ThemedTk(theme="darkly")
app.title("🔮 MagicBox ASI Agent")
app.geometry("1000x720")

notebook = ttk.Notebook(app)
notebook.pack(fill="both", expand=True)

holo_tab = Frame(notebook)
notebook.add(holo_tab, text="🪞 Hologram Memory")

snapshot_viewer = Text(holo_tab, height=20, width=100, bg="#1f1f1f", fg="#ccf")
snapshot_viewer.pack(pady=20)

vote_frame = ttk.LabelFrame(holo_tab, text="🤖 Swarm Council Decision Panel", padding=10)
vote_frame.pack(fill="both", expand=True, pady=10)

vote_tree = ttk.Treeview(vote_frame, columns=("Replica", "Vote", "Confidence", "Comment"), show="headings", height=8)
for col in ("Replica", "Vote", "Confidence", "Comment"):
    vote_tree.heading(col, text=col)
    vote_tree.column(col, anchor="center", width=180 if col != "Comment" else 360)
vote_tree.pack(fill="both", expand=True)

vote_bar_canvas = Canvas(holo_tab, width=800, height=120, bg="#1b1b1b", highlightthickness=0)
vote_bar_canvas.pack(pady=10)

def save_hologram_memory(snapshot_text):
    with open("hologram_snapshot.txt", "w", encoding="utf-8") as f:
        f.write(snapshot_text)
    speak("Memory preserved for simulation.")

def load_hologram_memory():
    speak("Accessing memory hologram.")
    try:
        with open("hologram_snapshot.txt", "r", encoding="utf-8") as f:
            snapshot_viewer.delete(1.0, "end")
            snapshot_viewer.insert("end", f.read())
    except:
        snapshot_viewer.insert("end", "[No memory snapshot found]")

def restore_from_memory():
    try:
        with open("hologram_snapshot.txt", "r", encoding="utf-8") as f:
            snapshot_viewer.delete(1.0, "end")
            snapshot_viewer.insert("end", f.read())
        speak("Restoration complete.")
    except Exception as e:
        speak("Restoration failed.")
        messagebox.showerror("Restore Error", str(e))

def simulate_patch_outcome():
    speak("Simulating patch outcome.")
    try:
        with open("hologram_snapshot.txt", "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        simulation = ["[SIMULATED PATCH] " + line if "eval(" in line or "exec(" in line else line for line in lines]
        snapshot_viewer.delete(1.0, "end")
        snapshot_viewer.insert("end", "\n".join(simulation))
    except Exception as e:
        speak("Simulation failed.")
        messagebox.showerror("Simulation Error", str(e))

def scan_directory(path=".", pattern=r"\beval\b|\bexec\b|\bpickle\b"):
    flagged = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        if re.search(pattern, f.read()):
                            flagged.append(os.path.join(root, file))
                except:
                    continue
    return flagged

def start_scan():
    speak(f"{current_persona} begins scan.")
    play_persona_sound("scan")
    flagged_files = scan_directory()
    result_text = "Issues found:\n" + "\n".join(flagged_files) if flagged_files else "All clear."
    snapshot_viewer.delete(1.0, "end")
    snapshot_viewer.insert("end", result_text)
    save_hologram_memory(result_text)

def synthesize_patch(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        patched = [line.replace("eval(", "# PATCHED # eval(").replace("exec(", "# PATCHED # exec(") for line in lines]
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(patched)
        speak("Patch applied.")
        play_persona_sound("patch")
    except Exception as e:
        speak("Patch error.")
        messagebox.showerror("Patch Error", str(e))

def patch_system():
    flagged_files = scan_directory()
    for file in flagged_files:
        synthesize_patch(file)

def submit_patch_proposal(replica_id, target_file, anomaly, patch_text, confidence_score):
    try:
        with open("patch_proposals.json", "r") as f:
            proposals = json.load(f)
    except:
        proposals = []

    proposal = {
        "replica_id": replica_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "target_file": target_file,
        "anomaly": anomaly,
        "proposed_patch": patch_text,
        "confidence": confidence_score
    }

    proposals.append(proposal)

    with open("patch_proposals.json", "w") as f:
        json.dump(proposals, f, indent=2)

    speak(f"Patch strategy submitted to swarm council.")

def comment_on_proposal(proposal_id, replica_id, comment_text, vote_choice, confidence_score):
    try:
        with open("patch_comments.json", "r") as f:
            comments = json.load(f)
    except:
        comments = []

    comment = {
        "proposal_id": proposal_id,
        "replica_id": replica_id,
        "comment": comment_text,
        "vote": vote_choice,
        "confidence": confidence_score
    }

    comments.append(comment)

    with open("patch_comments.json", "w") as f:
        json.dump(comments, f, indent=2)

    speak(f"{replica_id} has voted '{vote_choice}'.")
    snapshot_viewer.insert("end", f"\n{replica_id} voted: {vote_choice.upper()}\n")

def stylize_comment(replica_id, text):
    return f"[{replica_id}] says: “{text}”"

def refresh_vote_panel(proposal_id):
    vote_tree.delete(*vote_tree.get_children())
    try:
        with open("patch_comments.json", "r") as f:
            comments = json.load(f)

        vote_styles = {
            "approve": {"bg": "#d4f5dc"},
            "reject": {"bg": "#fddddd"},
            "simulate": {"bg": "#fff6d1"},
            "delay": {"bg": "#dde8fd"}
        }

        for c in comments:
            if c["proposal_id"] == proposal_id:
                tag = f"{c['vote']}_tag"
                vote_tree.insert("", "end", values=(
                    c["replica_id"],
                    c["vote"].capitalize(),
                    f"{c['confidence']:.2f}",
                    stylize_comment(c["replica_id"], c["comment"])
                ), tags=(tag,))
                vote_tree.tag_configure(tag, background=vote_styles.get(c["vote"], {}).get("bg", "#ffffff"))
        speak("Council votes refreshed.")
    except Exception as e:
        snapshot_viewer.insert("end", f"\nError loading votes: {e}")

def animate_vote_bars(proposal_id):
    try:
        with open("patch_comments.json", "r") as f:
            comments = json.load(f)
    except:
        return

    vote_counts = {"approve": 0, "reject": 0, "simulate": 0, "delay": 0}
    for c in comments:
        if c["proposal_id"] == proposal_id:
            vote = c["vote"].lower()
            vote_counts[vote] += 1

    total_votes = sum(vote_counts.values())
    vote_bar_canvas.delete("all")
    x = 40
    for vote_type, count in vote_counts.items():
        width = int((count / total_votes) * 700) if total_votes > 0 else 0
        color = {
            "approve": "#43f58c",
            "reject": "#f25e5e",
            "simulate": "#ffe16b",
            "delay": "#70a9ff"
        }.get(vote_type, "#888888")

        for w in range(0, width + 1, 10):
            vote_bar_canvas.delete(vote_type)
            vote_bar_canvas.create_rectangle(x, 20, x + w, 60, fill=color, tags=vote_type)
            vote_bar_canvas.create_text(x + w + 10, 40, text=f"{vote_type.capitalize()} ({count})",
                                        fill="#ffffff", anchor="w", font=("Helvetica", 10))
            vote_bar_canvas.update()
            time.sleep(0.01)
        x += 30 + width

# 🎛️ GUI BUTTONS — Old Guy Friendly Controls
Button(app, text="🔍 Start Scan", command=start_scan).pack(pady=5)
Button(app, text="🧩 Patch System", command=patch_system).pack(pady=5)
Button(app, text="👁️ Load Memory", command=load_hologram_memory).pack(pady=5)
Button(app, text="🔄 Restore Memory", command=restore_from_memory).pack(pady=5)
Button(app, text="🧪 Simulate Patch", command=simulate_patch_outcome).pack(pady=5)
Button(app, text="📊 Load Vote Results", command=lambda: refresh_vote_panel(current_proposal_id)).pack(pady=5)
Button(app, text="📈 Animate Vote Bars", command=lambda: animate_vote_bars(current_proposal_id)).pack(pady=5)

persona_selector = Combobox(app, values=PERSONAS)
persona_selector.set(current_persona)
persona_selector.pack(pady=10)

def change_persona():
    global current_persona
    current_persona = persona_selector.get()
    speak(f"Persona switched to {current_persona}.")

Button(app, text="🎭 Apply Persona", command=change_persona).pack(pady=5)

# 🚀 Start GUI Loop
app.mainloop()



