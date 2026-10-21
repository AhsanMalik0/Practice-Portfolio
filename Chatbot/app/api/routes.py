import json
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from app.config import MODEL
from app.services.rag_service import rag
from app.services.chatbot_service import ask, get_session, TooManyRequests, SORRY
from app.api.schemas import ChatIn

TEST_PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><title>Chatbot test</title>
<body style="font:15px system-ui;max-width:640px;margin:20px auto;padding:0 12px"><h3>Store chatbot test</h3>
<div id=log style="border:1px solid #ccc;border-radius:8px;padding:10px;height:55vh;overflow:auto;white-space:pre-wrap"></div>
<form id=f style="display:flex;gap:6px;margin-top:8px"><input id=q style="flex:1;padding:9px" placeholder="Ask something..." autocomplete=off><button>Send</button></form>
<script>const sid=Math.random().toString(36).slice(2),log=document.getElementById('log');
const add=(c,t)=>{const d=document.createElement('div');d.style.margin='6px 0';d.textContent=c+': '+t;log.appendChild(d);log.scrollTop=1e9;return d};
f.onsubmit=async e=>{e.preventDefault();const m=q.value.trim();if(!m)return;q.value='';add('You',m);const b=add('Bot','...');
const r=await fetch('/chat/stream?message='+encodeURIComponent(m)+'&session_id='+sid);
if(!r.ok){b.textContent='Bot: error '+r.status;return}b.textContent='Bot: ';const rd=r.body.getReader(),dc=new TextDecoder();
let buf='';for(;;){const{done,value}=await rd.read();if(done)break;buf+=dc.decode(value,{stream:true});
let i;while((i=buf.indexOf('\\n\\n'))>=0){const ev=buf.slice(0,i);buf=buf.slice(i+2);
if(ev.startsWith('data: '))b.textContent+=JSON.parse(ev.slice(6));log.scrollTop=1e9}}}</script>"""

def init_routes(app: FastAPI):
    @app.get("/", response_class=HTMLResponse)
    def home():
        return TEST_PAGE

    @app.get("/health")
    def health():
        return {"ok": True, "model": MODEL, "policy_chunks": rag().col.count_documents({})}

    @app.post("/chat")
    def chat(body: ChatIn):
        try:
            return ask(body.session_id, body.message)
        except TooManyRequests:
            raise HTTPException(429, "Too many messages, please wait a minute.")

    @app.get("/chat/stream")
    def chat_stream(message: str = Query(..., min_length=1, max_length=500), session_id: str = "anon"):
        try:
            s = get_session(session_id)
        except TooManyRequests:
            raise HTTPException(429, "Too many messages, please wait a minute.")

        def gen():
            try:
                for chunk in s["chat"].send_message_stream(message):
                    if chunk.text:
                        yield f"data: {json.dumps(chunk.text)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps(SORRY)}\n\n"
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")