class LoreJournal:
    def __init__(self):
        self.entries = []

    def record(self, avatar, ip, location):
        entry = f"🧿 {avatar} disrupted node at {location} — origin {ip}"
        self.entries.append(entry)
        print(f"[JOURNAL] {entry}")

    def export(self, filename="lore_journal.txt"):
        with open(filename, "w") as file:
            for line in self.entries:
                file.write(line + "\n")
        print("[JOURNAL] Entries exported.")

