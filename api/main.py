"""
FastAPI backend for Invest UP Chatbot
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from rag.retriever import InvestUPRetriever

app = FastAPI(
    title="Invest UP Chatbot API",
    description="RAG-powered chatbot for Uttar Pradesh investment information",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retriever
retriever = None


@app.on_event("startup")
async def startup_event():
    global retriever
    retriever = InvestUPRetriever()


class ChatRequest(BaseModel):
    """Chat request model"""
    question: str
    top_k: int = 5
    section_filter: Optional[str] = None


class Source(BaseModel):
    """Source document model"""
    url: str
    file: str
    section: str
    score: float


class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str
    sources: List[Source]
    query: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for asking questions about Invest UP

    - **question**: The question to ask
    - **top_k**: Number of documents to retrieve (default: 5)
    - **section_filter**: Optional filter by section (policy, sector, go, faq)
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        response = retriever.query(
            question=request.question,
            top_k=request.top_k,
            section_filter=request.section_filter
        )

        return ChatResponse(
            answer=response.answer,
            sources=[
                Source(
                    url=s["url"],
                    file=s["file"],
                    section=s["section"],
                    score=s["score"]
                )
                for s in response.sources
            ],
            query=response.query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sections")
async def get_sections():
    """Get available section filters"""
    return {
        "sections": [
            {"id": "policy", "name": "Policies"},
            {"id": "government_order", "name": "Government Orders"},
            {"id": "sector", "name": "Sector Information"},
            {"id": "faq", "name": "FAQs"},
            {"id": "scheme", "name": "Schemes"},
            {"id": "circular", "name": "Circulars"},
            {"id": "general", "name": "General Information"}
        ]
    }


# Example questions endpoint for UI
@app.get("/examples")
async def get_examples():
    """Get example questions"""
    return {
        "examples": [
            "What are the incentives for setting up an IT company in UP?",
            "What is the UP Semiconductor Policy 2024?",
            "How can I apply for the Gold Card scheme?",
            "What are the benefits for MSME units in Uttar Pradesh?",
            "What is the stamp duty exemption for industrial units?",
            "How do I get land for setting up a factory in UP?",
            "What are the incentives in the EV Manufacturing Policy?",
            "What approvals are required to start a food processing unit?"
        ]
    }


def run_server():
    """Run the API server"""
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    run_server()
