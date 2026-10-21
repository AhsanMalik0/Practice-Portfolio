import re
import numpy as np
from app.services.rag_service import rag

ORDERS = {
    "ORD-1001": {"email": "ali@example.com", "status": "Shipped", "items": "Wireless Earbuds x1", "total_pkr": 7999, "courier": "TCS", "tracking": "TCS771920", "ship_to": "Lahore", "estimated_delivery": "2026-10-15"},
    "ORD-1002": {"email": "sara@example.com", "status": "Processing", "items": "Running Shoes x1, Sports Socks x2", "total_pkr": 12499, "courier": "-", "tracking": "-", "ship_to": "Multan", "estimated_delivery": "2026-10-19"},
    "ORD-1003": {"email": "ali@example.com", "status": "Delivered", "items": "Backpack x1", "total_pkr": 5499, "courier": "Leopards", "tracking": "LP553018", "ship_to": "Karachi", "estimated_delivery": "2026-10-09"},
}
PRODUCTS = [
    {"name": "Wireless Earbuds", "price_pkr": 7999, "stock": 24, "tags": "earbuds headphones audio wireless"},
    {"name": "Running Shoes", "price_pkr": 10999, "stock": 0, "tags": "shoes running sneakers sports"},
    {"name": "Backpack", "price_pkr": 5499, "stock": 12, "tags": "bag backpack travel school"},
    {"name": "Smart Watch", "price_pkr": 15999, "stock": 7, "tags": "watch smartwatch fitness"},
    {"name": "Sports Socks", "price_pkr": 750, "stock": 80, "tags": "socks sports"},
]
ZONES = {
    "local": {"cities": ["multan"], "standard": 150, "express": 300, "days": "1-2 days"},
    "national": {"cities": ["lahore", "karachi", "islamabad", "rawalpindi", "faisalabad", "peshawar", "quetta", "sialkot"], "standard": 250, "express": 500, "days": "3-5 days"},
    "international": {"cities": ["dubai", "uk", "usa", "canada", "saudi arabia"], "standard": 1500, "express": 3000, "days": "7-14 days"},
}
STORE = {
    "name": "Demo Store", "hours": "Mon-Sat 10:00-20:00, Sun closed", "address": "12 Main Boulevard, Multan",
    "phone": "+92 300 0000000", "email": "support@demostore.example",
    "payment_methods": "Card, bank transfer, cash on delivery (domestic only)",
    "free_standard_delivery_over_pkr": 5000
}
FAILED_LOOKUPS = {}

def search_refund_policy(question: str) -> dict:
    """Search the store's refund, return, exchange and damaged-item policy documents."""
    hits = rag().search(question)
    if not hits:
        return {"excerpts": [], "note": "Nothing relevant found in the policy documents."}
    return {"excerpts": [{"text": h["text"], "source": h["source"]} for h in hits]}

def get_order_status(order_id: str, email: str) -> dict:
    m = re.search(r"(\d{3,})", order_id)
    oid = f"ORD-{m.group(1)}" if m else order_id.upper()
    if FAILED_LOOKUPS.get(oid, 0) >= 5:
        return {"error": "Too many failed attempts for this order. Please contact support."}
    order = ORDERS.get(oid)
    if not order or order["email"] != email.lower().strip():
        FAILED_LOOKUPS[oid] = FAILED_LOOKUPS.get(oid, 0) + 1
        return {"error": "Order number and email do not match our records."}
    FAILED_LOOKUPS.pop(oid, None)
    return {"order_id": oid, **{k: v for k, v in order.items() if k != "email"}}

def get_delivery_price(destination: str, weight_kg: float = 1.0, speed: str = "standard") -> dict:
    d = destination.lower()
    for zone, z in ZONES.items():
        if any(c in d for c in z["cities"]):
            extra = max(0, int(np.ceil(weight_kg - 1))) * 50
            price = z["express" if speed.lower().startswith("exp") else "standard"] + extra
            return {"destination": destination, "weight_kg": weight_kg, "speed": speed, "price_pkr": price,
                    "delivery_time": z["days"], "free_standard_over_pkr": 5000 if zone != "international" else None}
    return {"error": f"We do not have a delivery rate for '{destination}'.", "supported": [c for z in ZONES.values() for c in z["cities"]]}

def search_products(query: str) -> dict:
    words = [w for w in re.findall(r"[a-z]{3,}", query.lower())]
    hits = [p for p in PRODUCTS if any(w in f"{p['name']} {p['tags']}".lower() for w in words)]
    return {"matches": [{k: v for k, v in p.items() if k != "tags"} for p in (hits or [])], "all_product_names": [p["name"] for p in PRODUCTS]}

def get_store_info() -> dict:
    return STORE