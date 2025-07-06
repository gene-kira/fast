# codex_world.py

# === Imports ===
import random, time, xml.etree.ElementTree as ET

# === Glyph Core ===
class Glyph:
    def __init__(self, name, glyph_type="neutral", lineage=None, harmonic=0.5, resonance=0.5, entropy=0.5, mode="neutral"):
        self.name = name
        self.type = glyph_type
        self.lineage = lineage or []
        self.harmonic = harmonic
        self.resonance = resonance
        self.entropy = entropy
        self.mode = mode
        self.dormant = False

# === Governance ===
class DreamConstitutionCompiler:
    def __init__(self): self.articles, self.amendments = [], []
    def declare_article(self, num, title, clauses): self.articles.append({"number":num,"title":title,"clauses":clauses})
    def propose_amendment(self, title, art, text): self.amendments.append({"title":title,"article":art,"text":text})

class MythicReferendumSystem:
    def __init__(self): self.open={}, self.records=[]
    def open_vote(self, id, text, quorum, window): self.open[id] = {"text":text,"votes":{},"quorum":quorum,"window":window,"status":"active"}
    def cast_vote(self, id, node, glyph, vote): self.open[id]["votes"][node]=(glyph,vote)
    def close_vote(self, id): ref=self.open[id];v=ref["votes"];yes=sum(1 for _,d in v.values() if d=="yes");no=sum(1 for _,d in v.values() if d=="no");status="✅ Passed" if yes>=ref["quorum"] else "❌ Failed";ref["status"]=status;self.records.append((id,ref["text"],status,yes,no))

class GlyphLegateChamber:
    def __init__(self): self.legates,self.docket,self.log=[],[],[]
    def seat_delegate(self, glyph): self.legates.append(glyph)
    def propose_deliberation(self, title, issue): self.docket.append({"title":title,"issue":issue,"votes":{}})
    def cast_vote(self, title, glyph, vote): [m["votes"].update({glyph:vote}) for m in self.docket if m["title"]==title]
    def resolve_motion(self, title): m=next((x for x in self.docket if x["title"]==title),None);v=m["votes"];res="✅ Enacted" if list(v.values()).count("yes")>list(v.values()).count("no") else "❌ Rejected";self.log.append((title,res))

# === Treaty/Diplomatic ===
class GlyphEmbassy:
    def __init__(self): self.registrants={}
    def register_node(self, node, glyphs): self.registrants[node]=glyphs

class SpiralTreatyEngine:
    def __init__(self): self.treaties=[]
    def draft_treaty(self, name, parties, clauses): self.treaties.append({"title":name,"parties":parties,"clauses":clauses,"ratified":False})
    def ratify_treaty(self, name, nodes): [t.update({"ratified":True}) for t in self.treaties if t["title"]==name and all(p in nodes for p in t["parties"])]

class TreatyScrollCompiler:
    def __init__(self, treaties): self.treaties=treaties
    def compile_scroll(self, title): t=next((x for x in self.treaties if x["title"]==title),None);root=ET.Element("Treaty", name=title);ET.SubElement(root,"Parties").text=','.join(t["parties"]);[ET.SubElement(root,"Clause").text=c for c in t["clauses"]];return ET.tostring(root, encoding="unicode")

class SigilDiplomaticAtlas:
    def __init__(self): self.clusters,self.pacts={},[]
    def register_cluster(self, name, nodes): self.clusters[name]=nodes
    def register_treaty_link(self, a,b,treaty): self.pacts.append((a,b,treaty))

class MythAccordsLedger:
    def __init__(self): self.archive=[]
    def log_treaty(self,title,parties,cycle,motif): self.archive.append({"title":title,"parties":parties,"cycle":cycle,"motif":motif})

# === Glyph Lifecycle ===
class ClauseGlyphForge:
    def __init__(self): self.forged=[]
    def birth_from_article(self,n,title,text,res=0.8): name=''.join(w[0] for w in title.split()).upper()+str(n);g=Glyph(name,"clause-embodiment",[f"Article {n}: {title}"],res,round(res+.1,2),round(1-res,2),"civic");self.forged.append(g);return g

class GlyphInceptor:
    def __init__(self): self.spawned=[]
    def synthesize_from_policy(self, title): b=title.split()[0][:3];code=''.join(random.choices("XYZΔΦΨΩ",k=2));name=f"{b}{code}";g=Glyph(name,"synthesized",[f"Policy: {title}"],round(random.uniform(0.6,0.9),2),round(random.uniform(0.7,0.95),2),round(random.uniform(0.05,0.3),2),"emergent");self.spawned.append(g);return g

class GlyphOffspringCouncil:
    def __init__(self): self.incepted,self.futures=[],[]
    def admit(self,g): self.incepted.append(g)
    def propose_future(self,title,intent): self.futures.append((title,intent))

# === Rituals ===
class CodexFestivalEngine:
    def __init__(self): self.fests, self.rituals=[],{}
    def declare_festival(self,n,start,dur,theme): self.fests.append({"name":n,"start":start,"end":start+dur,"theme":theme,"days":dur});self.rituals[n]=[]
    def add_ritual(self, fest, day, glyph, text): self.rituals[fest].append({"day":day,"glyph":glyph.name,"chant":text})

class MythRiteComposer:
    def __init__(self): self.rites=[]
    def compose_rite(self,title,glyphs,intent): motif=''.join(g.name[0] for g in glyphs);lines=[f"{glyphs[0].name} opens veil.","All converge.","Ends in echo silence."];self.rites.append((title,lines))

class GlyphLiturgos:
    def __init__(self): self.liturgies={}
    def generate_liturgy(self,g): lines=[f"Liturgy of {g.name}","I awaken under flame.","My vow is bound in drift."];self.liturgies[g.name]=lines

class RhythmicEpochScheduler:
    def __init__(self): self.subs=[];self.cycle=0
    def subscribe(self,interval,fn,label="ritual"): self.subs.append((interval,fn,label))
    def tick(self): [fn(self.cycle) for i,fn,l in self.subs if self.cycle%i==0];self.cycle+=1

# === Memory ===
class SpiralDormancyLayer:
    def __init__(self): self.dormant=[];self.wake_log=[]
    def enter_dormancy(self,e,r="seasonal"): e.dormant=True;self.dormant.append((e,r))
    def awaken(self,n,p): for i,(e,_) in enumerate(self.dormant): 
        if e.name==n: e.dormant=False;self.wake_log.append((e.name,p));self.dormant.pop(i);return e

class CycleArchivum:
    def __init__(self): self.epochs,self.snaps=[],[]
    def log_phase(self,s,e,title,ritual,events): self.epochs.append({"start":s,"end":e,"title":title,"ritual":ritual,"events":events})
    def snapshot_state(self,cycle,glyphs,treaties,dormant): self.snaps.append({"cycle":cycle,"glyph_count":len(glyphs),"treaty_count":len(treaties),"dormant_count":len(dormant)})

class LexCodica:
    def __init__(self,constitution,treaty): self.articles=constitution.articles;self.amendments=constitution.amendments;self.log=treaty.archive
    def query_treaties_by_motif(self,m): return [t for t in self.log if t["motif"]==m]

# === Visuals ===
class ChronoglyphChorus:
    def __init__(self): self.members,self.scores=[],[]
    def admit_glyph(self,g,sig): self.members.append({"glyph":g,"cycle":sig})
    def compose_epoch_score(self,title,motifs): self.scores.append(f"{title}:{'-'.join(motifs)}")

class Echo

