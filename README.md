# AI-Powered College Admission Assistant

> Built with **IBM Watsonx Granite** · **LangChain** · **ChromaDB** · **FastAPI** · **React**

An intelligent RAG-based chatbot that answers student admission queries by retrieving information from official admission documents (PDFs, brochures, FAQs) and generating accurate responses using IBM Granite LLM — with enterprise-grade guardrails.

---

## Features
- 🤖 RAG Pipeline (ChromaDB + Sentence Transformers + IBM Granite)
- 🛡️ 15-layer Guardrails (Injection, Toxicity, PII, Confidence, Audit)
- 📄 PDF / DOCX document upload & indexing
- 💬 Student Chat Interface (React + Tailwind)
- 🔐 JWT Auth + Role-Based Access Control
- 📊 Admin Dashboard & Analytics
- 📝 Full Audit Logging

## Tech Stack
| Layer | Technology |
|---|---|
| Frontend | React.js, Tailwind CSS, Axios |
| Backend | FastAPI (Python) |
| LLM | IBM Watsonx Granite (`ibm/granite-13b-chat-v2`) |
| RAG | LangChain, ChromaDB, Sentence Transformers |
| Database | MongoDB |
| Document Processing | pdfplumber, PyPDF2 |

## Quick Start

```bash
# Backend
cd backend
cp .env.example .env        # Fill in your IBM Watsonx credentials
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm start
```

## Environment Variables
See `backend/.env.example` for all required variables including:
- `WATSONX_API_KEY`
- `WATSONX_PROJECT_ID`
- `MONGODB_URL`
- `JWT_SECRET_KEY`

## Project Structure
```
college-admission-assistant/
├── backend/
│   ├── app/
│   │   ├── api/routes/       # auth, chat, documents, admin
│   │   ├── core/             # config, security
│   │   ├── db/               # MongoDB connection
│   │   ├── models/           # Pydantic schemas
│   │   └── services/
│   │       ├── rag/          # vector_store, llm_service, pipeline
│   │       ├── guardrails/   # engine (15 guardrails)
│   │       └── document/     # processor (PDF/DOCX extraction)
│   └── requirements.txt
└── frontend/                 # React app (coming soon)
```

## License
MIT
