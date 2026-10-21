import time
from google import genai
from google.genai import types
from app.config import MODEL, logging, GEMINI_API_KEY
from app.tools.store_tools import (
    STORE, search_refund_policy, get_order_status, get_delivery_price, search_products, get_store_info
)

log = logging.getLogger("bot")
client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM = f"""You are the customer support assistant for {STORE['name']}, an online store. Be friendly and brief (1 to 4 short sentences).
Rules:
- For refunds/returns, call search_refund_policy and answer ONLY from the excerpts it returns. If they do not cover the question, say you are not sure and give {STORE['email']}.
- For order questions, you need both the order number and the order email. Ask for what is missing, then call get_order_status. Never reveal order details without a successful lookup.
- For delivery cost or time, call get_delivery_price (ask for the destination if missing). For products call search_products. For hours/contact/payments call get_store_info.
- Never invent prices, dates, policies or order data. Only use tool results. Currency is PKR.
- Politely decline anything unrelated to the store. Reply in the same language the customer writes in."""

def make_config():
    return types.GenerateContentConfig(
        system_instruction=SYSTEM, 
        temperature=0.2, 
        max_output_tokens=800,
        tools=[search_refund_policy, get_order_status, get_delivery_price, search_products, get_store_info]
    )

SESSIONS = {}

class TooManyRequests(Exception):
    pass

def get_session(sid):
    now = time.time()
    for k in [k for k, v in SESSIONS.items() if now - v["t"] > 1800]:
        del SESSIONS[k]
    s = SESSIONS.get(sid)
    if not s:
        s = SESSIONS[sid] = {"chat": client.chats.create(model=MODEL, config=make_config()), "hits": []}
    s["t"] = now
    s["hits"] = [h for h in s["hits"] if now - h < 60]
    if len(s["hits"]) >= 20:
        raise TooManyRequests()
    s["hits"].append(now)
    return s

SORRY = f"Sorry, I am having trouble right now. Please try again or email {STORE['email']}."

def ask(sid, text):
    t0 = time.perf_counter()
    try:
        reply = get_session(sid)["chat"].send_message(text).text or SORRY
    except TooManyRequests:
        raise
    except Exception as e:
        log.error("Gemini error: %s", e)
        reply = SORRY
    return {"reply": reply, "ms": round((time.perf_counter() - t0) * 1000)}