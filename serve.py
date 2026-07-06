"""
serve.py -- Pip WebUI: a tiny zero-dependency web chat app for the homemade
ax0/ax1 "Pip" models, with a built-in MODEL LOADER.

    python3 serve.py                      # scan ./models, open http://localhost:8000
    python3 serve.py --models-dir models --model models/pip4.2.npz --port 8000

Pure Python standard library (http.server) + numpy. No Flask, no Gradio. Drop any
Pip checkpoint (a *.npz saved by save_ax0 / convert_mlx_to_numpy) into the models/
folder and pick it from the dropdown in the header -- the server loads it live and
you keep chatting. The reply streams token-by-token so you watch Pip type.
"""

import argparse
import glob
import json
import os
import threading
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from infer import Chatbot

# --- global model state ------------------------------------------------------
BOT = None            # current Chatbot (or None if nothing loaded)
CURRENT = None        # current model name (basename)
MODELS_DIR = "models"
LOAD_LOCK = threading.Lock()   # serialize model loads
GEN_LOCK = threading.Lock()    # one reply streams at a time (shared recurrent state)


def list_models():
    """All *.npz in the models dir, newest first, with sizes."""
    out = []
    for p in glob.glob(os.path.join(MODELS_DIR, "*.npz")):
        if os.path.basename(p).startswith("._"):
            continue
        out.append({"name": os.path.basename(p),
                    "size_mb": round(os.path.getsize(p) / 1e6)})
    out.sort(key=lambda m: m["name"])
    return out


def bot_info():
    if BOT is None:
        return None
    m = BOT.model
    return {
        "codename": BOT.codename,
        "current": CURRENT,
        "arch": BOT.arch,
        "params": int(m.num_params()),
        "meta": f"{m.cfg.n_layer}L/{m.cfg.n_head}H/{m.cfg.n_embd}d · "
                f"ctx {m.cfg.block_size} · {m.num_params()/1e6:.1f}M params",
    }


def load_model(name):
    """Load a model by basename from the models dir. Returns bot_info() or raises."""
    global BOT, CURRENT
    safe = os.path.basename(name)                 # no path traversal
    path = os.path.join(MODELS_DIR, safe)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{safe} not found in {MODELS_DIR}/")
    with LOAD_LOCK:
        bot = Chatbot(path)                       # may take a few seconds for big models
        BOT, CURRENT = bot, safe
    return bot_info()


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pip WebUI</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--edge:#262b36;--ink:#e6e9ef;--mut:#8b93a3;
--me:#1f6feb;--bot:#222834;--accent:#3fb950;--warn:#d29922}
*{box-sizing:border-box}
body{margin:0;font:15px/1.6 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
background:var(--bg);color:var(--ink);height:100vh;display:flex;flex-direction:column}
header{padding:12px 18px;border-bottom:1px solid var(--edge);background:var(--panel);
display:flex;align-items:center;gap:12px;flex-wrap:wrap}
header .dot{width:9px;height:9px;border-radius:50%;background:var(--mut);box-shadow:0 0 8px var(--mut);flex:none}
header .dot.on{background:var(--accent);box-shadow:0 0 8px var(--accent)}
header .dot.busy{background:var(--warn);box-shadow:0 0 8px var(--warn)}
header h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.3px}
.loader{display:flex;gap:6px;align-items:center}
#model{background:#0c0e12;border:1px solid var(--edge);color:var(--ink);border-radius:8px;
padding:6px 8px;font:inherit;font-size:13px;outline:none;max-width:240px}
#model:focus{border-color:var(--me)}
#reload{background:#0c0e12;border:1px solid var(--edge);color:var(--mut);border-radius:8px;
padding:6px 9px;cursor:pointer;font-size:14px;line-height:1}
#reload:hover{color:var(--ink);border-color:var(--me)}
header .meta{color:var(--mut);font-size:12px;margin-left:auto;font-family:ui-monospace,Menlo,monospace;text-align:right}
#log{flex:1;overflow-y:auto;padding:22px;display:flex;flex-direction:column;gap:14px;max-width:820px;width:100%;margin:0 auto}
.row{display:flex;gap:10px;align-items:flex-end}
.row.me{flex-direction:row-reverse}
.bubble{padding:10px 14px;border-radius:14px;max-width:74%;white-space:pre-wrap;word-wrap:break-word}
.me .bubble{background:var(--me);border-bottom-right-radius:4px}
.bot .bubble{background:var(--bot);border:1px solid var(--edge);border-bottom-left-radius:4px}
.who{font-size:11px;color:var(--mut);margin:0 6px}
.sys{align-self:center;color:var(--mut);font-size:12px;font-style:italic}
.cursor::after{content:"\\2588";animation:blink 1s steps(2) infinite;color:var(--accent)}
@keyframes blink{50%{opacity:0}}
.hint{color:var(--mut);font-size:12.5px;text-align:center;margin:6px 0 0}
footer{border-top:1px solid var(--edge);background:var(--panel);padding:12px 18px}
.bar{display:flex;gap:10px;max-width:820px;margin:0 auto;align-items:center}
#msg{flex:1;background:#0c0e12;border:1px solid var(--edge);color:var(--ink);
border-radius:10px;padding:11px 14px;font:inherit;outline:none}
#msg:focus{border-color:var(--me)}
button.send{background:var(--me);color:#fff;border:0;border-radius:10px;padding:11px 18px;
font:inherit;font-weight:600;cursor:pointer}button.send:disabled{opacity:.5;cursor:default}
.ctl{display:flex;gap:16px;max-width:820px;margin:8px auto 0;color:var(--mut);font-size:12px;align-items:center;font-family:ui-monospace,Menlo,monospace}
.ctl label{display:flex;gap:7px;align-items:center}
.ctl input[type=range]{accent-color:var(--me)}
.ctl a{color:var(--mut);margin-left:auto;cursor:pointer;text-decoration:underline}
.bubble pre{background:#0c0e12;border:1px solid var(--edge);border-radius:8px;padding:8px 10px;overflow-x:auto;white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:13px;margin:6px 0}
.bubble code{background:#0c0e12;border-radius:4px;padding:1px 5px;font-family:ui-monospace,Menlo,monospace;font-size:.9em}
.bubble pre code{background:none;padding:0}
.bubble h2,.bubble h3{margin:6px 0 3px;font-weight:600}
.bubble ul,.bubble ol{margin:4px 0;padding-left:20px}
.bubble a{color:var(--me)}
</style></head><body>
<header><span class="dot" id="dot"></span><h1 id="codename">Pip WebUI</h1>
<div class="loader"><select id="model" title="Load a Pip model"></select>
<button id="reload" title="Rescan the models folder">&#8635;</button></div>
<span class="meta" id="meta">no model loaded</span></header>
<div id="log"><p class="hint" id="hint">Pick a model from the dropdown to start chatting.</p></div>
<footer>
<div class="bar">
<input id="msg" placeholder="Message Pip…" autocomplete="off" disabled>
<button class="send" id="send" disabled>Send</button>
</div>
<div class="ctl">
<label>temp <input id="temp" type="range" min="0.2" max="1.3" step="0.1" value="0.4"><span id="tv">0.4</span></label>
<label>top-k <input id="topk" type="range" min="1" max="60" step="1" value="30"><span id="kv">30</span></label>
<a id="clear">clear chat</a>
</div>
</footer>
<script>
const log=document.getElementById('log'),msg=document.getElementById('msg'),send=document.getElementById('send');
const temp=document.getElementById('temp'),topk=document.getElementById('topk');
const modelSel=document.getElementById('model'),reload=document.getElementById('reload');
const dot=document.getElementById('dot'),codename=document.getElementById('codename'),meta=document.getElementById('meta'),hint=document.getElementById('hint');
temp.oninput=()=>tv.textContent=temp.value; topk.oninput=()=>kv.textContent=topk.value;
let turns=[], loaded=false, busy=false;

function add(who,text){const r=document.createElement('div');r.className='row '+who;
 r.innerHTML=`<div class="who">${who==='me'?'You':'Pip'}</div><div class="bubble"></div>`;
 r.querySelector('.bubble').textContent=text;log.appendChild(r);log.scrollTop=log.scrollHeight;
 return r.querySelector('.bubble');}
function sys(text){const d=document.createElement('div');d.className='sys';d.textContent=text;log.appendChild(d);log.scrollTop=log.scrollHeight;}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function md(t){t=esc(t);
 t=t.replace(/```([\\s\\S]*?)```/g,(m,c)=>'<pre>'+c.replace(/^\\n+|\\n+$/g,'')+'</pre>');
 t=t.replace(/`([^`\\n]+)`/g,'<code>$1</code>');
 t=t.replace(/^\\s{0,3}#{1,6}\\s+(.*)$/gm,'<h3>$1</h3>');
 t=t.replace(/\\*\\*([^*]+)\\*\\*/g,'<b>$1</b>');
 t=t.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g,'<a href="$2" target="_blank">$1</a>');
 return t;}
function setBusy(b,label){busy=b;dot.className='dot '+(b?'busy':(loaded?'on':''));
 send.disabled=b||!loaded;msg.disabled=b||!loaded;modelSel.disabled=b;
 if(label)meta.textContent=label;}
function applyInfo(info){
 if(info){loaded=true;codename.textContent=info.codename;meta.textContent=info.meta;
  dot.className='dot on';send.disabled=false;msg.disabled=false;msg.focus();
  if(hint)hint.textContent='Loaded '+info.codename+'. Say hi — keep it simple, it\\'s tiny.';}
 else{loaded=false;codename.textContent='Pip WebUI';meta.textContent='no model loaded';dot.className='dot';}
}
async function refreshModels(){
 try{const j=await (await fetch('/api/models')).json();
  modelSel.innerHTML='';
  if(!j.models.length){const o=document.createElement('option');o.textContent='(no models in models/)';o.value='';modelSel.appendChild(o);}
  j.models.forEach(m=>{const o=document.createElement('option');o.value=m.name;o.textContent=m.name+'  ('+m.size_mb+' MB)';if(m.name===j.current)o.selected=true;modelSel.appendChild(o);});
  applyInfo(j.info);
 }catch(e){meta.textContent='error listing models';}
}
async function loadModel(name){
 if(!name)return; setBusy(true,'loading '+name+'…');
 try{const j=await (await fetch('/api/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})})).json();
  if(j.error){sys('load failed: '+j.error);setBusy(false);}
  else{applyInfo(j.info);sys('✓ loaded '+j.info.codename+' ('+j.info.meta+')');setBusy(false);}
 }catch(e){sys('load error: '+e);setBusy(false);}
}
modelSel.onchange=e=>loadModel(e.target.value);
reload.onclick=refreshModels;
document.getElementById('clear').onclick=()=>{turns=[];[...log.querySelectorAll('.row,.sys')].forEach(e=>e.remove());};
async function go(){const m=msg.value.trim();if(!m||busy||!loaded)return;
 msg.value='';send.disabled=true;msg.disabled=true;
 add('me',m);const b=add('bot','');b.classList.add('cursor');
 let reply="";
 try{
  const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({turns,message:m,temp:parseFloat(temp.value),topk:parseInt(topk.value)})});
  const reader=res.body.getReader();const dec=new TextDecoder();
  for(;;){const {done,value}=await reader.read();if(done)break;
   reply+=dec.decode(value,{stream:true});b.textContent=reply;log.scrollTop=log.scrollHeight;}
 }catch(e){reply='(connection error)';b.textContent=reply;}
 b.classList.remove('cursor');
 b.innerHTML=md(reply);
 turns.push({role:'user',text:m},{role:'pip',text:reply.trim()});
 if(turns.length>16)turns=turns.slice(-12);
 send.disabled=false;msg.disabled=false;msg.focus();}
send.onclick=go;msg.addEventListener('keydown',e=>{if(e.key==='Enter')go();});
refreshModels();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self):
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._html()
        elif self.path == "/api/models":
            self._json({"models": list_models(), "current": CURRENT, "info": bot_info()})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/load":
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            try:
                info = load_model(req.get("name", ""))
                self._json({"ok": True, "info": info})
            except Exception as e:
                self._json({"error": str(e)}, code=400)
        elif self.path == "/chat":
            self._chat()
        else:
            self.send_error(404)

    def _chat(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        bot = BOT
        turns = [(t.get("role"), t.get("text", "")) for t in req.get("turns", [])]
        temp = float(req.get("temp", 0.4))
        topk = int(req.get("topk", 30))
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        if bot is None:
            self.wfile.write("(no model loaded — pick one from the dropdown)".encode("utf-8"))
            return
        try:
            with GEN_LOCK:
                for piece in bot.stream(turns, req.get("message", ""), temp, topk):
                    self.wfile.write(piece.encode("utf-8"))
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    global MODELS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models", help="folder to scan for *.npz Pip models")
    ap.add_argument("--model", default=None, help="model to load at startup (path or name in models-dir)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    MODELS_DIR = args.models_dir
    os.makedirs(MODELS_DIR, exist_ok=True)

    # choose an initial model: --model, else the only/first one in the folder
    initial = args.model
    if initial is None:
        found = list_models()
        if len(found) == 1:
            initial = found[0]["name"]
    if initial:
        try:
            if os.path.dirname(initial):          # a path outside models-dir -> copy name in
                import shutil
                dst = os.path.join(MODELS_DIR, os.path.basename(initial))
                if os.path.abspath(initial) != os.path.abspath(dst):
                    shutil.copy(initial, dst)
                initial = os.path.basename(dst)
            load_model(initial)
            print(f"  loaded {CURRENT}  [{BOT.arch}]  ({BOT.model.num_params():,} params)")
        except Exception as e:
            print(f"  (could not load {initial}: {e})")

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    n = len(list_models())
    print(f"  Pip WebUI  |  {n} model(s) in {MODELS_DIR}/")
    print(f"  serving on  http://localhost:{args.port}   (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  bye")


if __name__ == "__main__":
    main()
