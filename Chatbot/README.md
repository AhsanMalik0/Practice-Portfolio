#  E-Commerce Customer Support Chatbot API (Gemini RAG)

A production-ready, modular FastAPI backend that powers an intelligent e-commerce customer support assistant. Driven by Google's **Gemini API**, this project utilizes **Function Calling (Tools)** and **Retrieval-Augmented Generation (RAG)** backed by **MongoDB Vector Search** to provide real-time order tracking, shipping estimates, product catalog search, and policy answers without hallucinating store information.

---

##  Features

- **Dynamic RAG Architecture:** Reads `.txt`, `.md`, and `.pdf` files from `policy_docs/`, generates vector embeddings via Gemini, and indexes them in MongoDB.
- **Agentic Function Calling:** Gemini dynamically executes tool calls to query live store data (Orders, Products, Delivery Rates, Store Details, Policy Excerpts).
- **Streaming Responses (SSE):** Built-in Server-Sent Events endpoint (`/chat/stream`) for real-time word-by-word streaming in frontends.
- **Session Management:** Tracks conversation context and enforces rate-limiting per customer session.
- **Enterprise-Ready Modular Architecture:** Clean separation of concerns into configuration, services, tools, and API layers.
- **Security-First Configuration:** Zero hardcoded API keys—environment variable management using `.env`.

---

##  Repository Structure

```text
store_chatbot_backend/
├── .env.example            # Environment template file
├── .gitignore              # Ignores local credentials and caches
├── requirements.txt        # Python package dependencies
├── main.py                 # Application entry point
├── policy_docs/            # Store policy documents (.txt, .md, .pdf)
└── app/
    ├── config.py           # Environment variables & logger config
    ├── database.py         # MongoDB client initialization
    ├── api/
    │   ├── routes.py       # FastAPI endpoints & web interface
    │   └── schemas.py      # Pydantic request/response models
    ├── services/
    │   ├── rag_service.py  # Document chunking, embedding & vector search
    │   └── chatbot_service.py # Gemini session management & prompt setup
    └── tools/
        └── store_tools.py  # Python functions exposed as Gemini Agent tools