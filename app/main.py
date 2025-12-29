"""
Invest UP Chatbot - Flask Application
RAG-powered chatbot for UP Investment ecosystem
Enhanced with Hybrid Search + Re-ranking + Conversation Memory
"""
import os
import sys
import re
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, send_file, session

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    EMBEDDING_DIMENSIONS
)

from pinecone import Pinecone
from openai import AzureOpenAI
import tiktoken

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "invest-up-chatbot-secret-key-2024")

# Conversation memory storage (in-memory, keyed by session_id)
# In production, use Redis or a database
conversation_store = defaultdict(list)
MAX_CONVERSATION_TURNS = 30  # Keep last 30 Q&A pairs for longer context
CONVERSATION_EXPIRY_HOURS = 24

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

llm_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Embedding client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "")

if AZURE_EMBEDDING_DEPLOYMENT:
    embedding_client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2023-05-15",
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
    embedding_model = AZURE_EMBEDDING_DEPLOYMENT
else:
    from openai import OpenAI
    embedding_client = OpenAI(api_key=OPENAI_API_KEY)
    embedding_model = "text-embedding-3-large"


def get_embedding(text: str) -> list:
    """Generate embedding for query text"""
    response = embedding_client.embeddings.create(
        input=text,
        model=embedding_model,
        dimensions=EMBEDDING_DIMENSIONS
    )
    return response.data[0].embedding


# Initialize hybrid retriever (lazy loading)
_retriever = None

def get_hybrid_retriever():
    """Get or create hybrid retriever instance"""
    global _retriever
    if _retriever is None:
        from retriever import get_retriever
        _retriever = get_retriever(index, get_embedding)
    return _retriever


def search_documents(query: str) -> list:
    """
    Search using enhanced hybrid retrieval with all improvements:
    1. Query rewriting - Complex queries split into focused sub-queries
    2. Query processing - Spelling correction, Hindi-English handling, synonym expansion
    3. Dynamic top_k - Adjusts based on query complexity (5-8 final results)
    4. Hybrid search - Vector (Pinecone) + BM25 keyword matching with RRF fusion
    5. Multi-query fusion - Results from all query variants merged with boosting
    6. Cross-encoder re-ranking - Final semantic relevance scoring

    Returns list of most relevant document chunks.
    """
    try:
        retriever = get_hybrid_retriever()
        results = retriever.search(query)
        return results
    except Exception as e:
        print(f"[Search] Hybrid search failed, falling back to vector-only: {e}")
        # Fallback to simple vector search
        return search_documents_fallback(query, top_k=8)


def search_documents_fallback(query: str, top_k: int = 5) -> list:
    """Fallback to simple vector search if hybrid fails"""
    query_embedding = get_embedding(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return [
        {
            "id": match.id,
            "score": match.score,
            "text": match.metadata.get("text", ""),
            "source_url": match.metadata.get("source_url", ""),
            "source_file": match.metadata.get("source_file", ""),
            "portal": match.metadata.get("portal", ""),
            "section": match.metadata.get("section", ""),
            "doc_type": match.metadata.get("doc_type", ""),
            "chunk_index": match.metadata.get("chunk_index", 0),
            "total_chunks": match.metadata.get("total_chunks", 1)
        }
        for match in results.matches
    ]


def smart_truncate(text: str, max_chars: int = 2500) -> str:
    """
    Truncate text at sentence boundary instead of cutting mid-sentence.
    Increased limit to 2500 chars and ensures we end at a complete sentence.
    """
    if len(text) <= max_chars:
        return text

    # Find the last sentence boundary before max_chars
    truncated = text[:max_chars]

    # Look for sentence endings: . ! ? and Hindi danda
    sentence_endings = ['.', '!', '?', '।', '॥']

    # Find the last sentence ending
    last_end = -1
    for ending in sentence_endings:
        pos = truncated.rfind(ending)
        if pos > last_end:
            last_end = pos

    if last_end > max_chars * 0.5:  # At least 50% of content
        return truncated[:last_end + 1]

    # If no good sentence boundary, try to end at a paragraph
    last_para = truncated.rfind('\n\n')
    if last_para > max_chars * 0.5:
        return truncated[:last_para]

    # Last resort: end at last space
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return truncated[:last_space] + "..."

    return truncated + "..."


def get_conversation_history(session_id: str) -> list:
    """Get conversation history for a session"""
    if session_id not in conversation_store:
        return []
    return conversation_store[session_id][-MAX_CONVERSATION_TURNS:]


def add_to_conversation(session_id: str, query: str, answer: str):
    """Add a Q&A pair to conversation history"""
    conversation_store[session_id].append({
        "query": query,
        "answer": answer[:500],  # Store truncated answer to save memory
        "timestamp": datetime.now().isoformat()
    })
    # Keep only last N turns
    if len(conversation_store[session_id]) > MAX_CONVERSATION_TURNS:
        conversation_store[session_id] = conversation_store[session_id][-MAX_CONVERSATION_TURNS:]


def format_conversation_context(history: list) -> str:
    """Format conversation history for the prompt"""
    if not history:
        return ""

    context_parts = ["**Previous conversation:**"]
    for turn in history[-3:]:  # Last 3 turns only
        context_parts.append(f"User: {turn['query']}")
        context_parts.append(f"Assistant: {turn['answer'][:200]}...")

    return "\n".join(context_parts) + "\n\n"


def verify_citations(answer: str, num_sources: int, doc_names: list) -> str:
    """
    Verify and fix citations in the LLM response.
    - Ensures [Document X] references are valid (X <= num_sources)
    - Replaces invalid citations with valid document names
    """
    if num_sources == 0:
        return answer

    # Find all citation patterns: [Document X], [Doc X], [Source X], [X]
    citation_patterns = [
        (r'\[Document\s*(\d+)\]', 'Document'),
        (r'\[Doc\s*(\d+)\]', 'Doc'),
        (r'\[Source\s*(\d+)\]', 'Source'),
    ]

    fixed_answer = answer
    invalid_citations = []

    for pattern, prefix in citation_patterns:
        matches = re.findall(pattern, fixed_answer, re.IGNORECASE)
        for match in matches:
            doc_num = int(match)
            if doc_num > num_sources or doc_num < 1:
                invalid_citations.append(f"[{prefix} {doc_num}]")
                # Replace with first valid document name
                if doc_names:
                    replacement = f"[{doc_names[0]}]"
                else:
                    replacement = f"[Document 1]"
                fixed_answer = re.sub(
                    rf'\[{prefix}\s*{doc_num}\]',
                    replacement,
                    fixed_answer,
                    flags=re.IGNORECASE
                )

    if invalid_citations:
        print(f"[Citation] Fixed invalid citations: {invalid_citations}")

    return fixed_answer


def check_answer_grounding(answer: str, sources: list, threshold: float = 0.3) -> dict:
    """
    Check if the answer is grounded in the source documents.
    Returns grounding score and warning if answer may not be well-grounded.

    Uses simple keyword overlap for efficiency.
    """
    if not sources or not answer:
        return {"grounded": False, "score": 0.0, "warning": "No sources available"}

    # Extract key terms from answer (words > 4 chars, excluding common words)
    common_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'has', 'have',
        'will', 'your', 'from', 'they', 'been', 'would', 'there', 'their', 'what',
        'about', 'which', 'when', 'make', 'like', 'time', 'very', 'after', 'with',
        'also', 'some', 'than', 'into', 'could', 'other', 'this', 'that', 'these',
        'those', 'being', 'more', 'most', 'such', 'only', 'over', 'under', 'through',
        'document', 'based', 'following', 'according', 'mentioned', 'provided',
        'information', 'details', 'available', 'specific', 'include', 'includes'
    }

    # Tokenize answer
    answer_words = set(
        word.lower() for word in re.findall(r'\b[a-zA-Z]{4,}\b', answer)
        if word.lower() not in common_words
    )

    if not answer_words:
        return {"grounded": True, "score": 1.0, "warning": None}

    # Combine all source texts
    source_text = " ".join(s.get("text", "") for s in sources)
    source_words = set(
        word.lower() for word in re.findall(r'\b[a-zA-Z]{4,}\b', source_text)
    )

    # Calculate overlap
    overlap = answer_words.intersection(source_words)
    score = len(overlap) / len(answer_words) if answer_words else 0

    grounded = score >= threshold
    warning = None

    if not grounded:
        warning = "This answer may contain information not found in the source documents. Please verify with official sources."

    return {
        "grounded": grounded,
        "score": round(score, 2),
        "warning": warning
    }


def get_specific_error_message(error: Exception, context: str = "query") -> dict:
    """
    Generate user-friendly error messages based on error type.
    Returns dict with 'message' for user and 'details' for logging.
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    # API/Rate limit errors
    if "rate" in error_str or "429" in error_str:
        return {
            "message": "The service is currently experiencing high demand. Please wait a moment and try again.",
            "details": f"Rate limit error: {error}",
            "code": "RATE_LIMIT"
        }

    # Authentication errors
    if "auth" in error_str or "401" in error_str or "403" in error_str:
        return {
            "message": "There was an authentication issue with the service. Please contact support.",
            "details": f"Auth error: {error}",
            "code": "AUTH_ERROR"
        }

    # Timeout errors
    if "timeout" in error_str or "timed out" in error_str:
        return {
            "message": "The request took too long to process. Please try a shorter or more specific query.",
            "details": f"Timeout error: {error}",
            "code": "TIMEOUT"
        }

    # Connection errors
    if "connection" in error_str or "network" in error_str:
        return {
            "message": "Unable to connect to the service. Please check your internet connection and try again.",
            "details": f"Connection error: {error}",
            "code": "CONNECTION_ERROR"
        }

    # Token/context length errors
    if "token" in error_str or "context" in error_str or "length" in error_str:
        return {
            "message": "Your query or the matching documents are too long. Please try a more specific question.",
            "details": f"Token limit error: {error}",
            "code": "TOKEN_LIMIT"
        }

    # Pinecone/Vector DB errors
    if "pinecone" in error_str or "index" in error_str:
        return {
            "message": "There was an issue searching the document database. Please try again in a moment.",
            "details": f"Database error: {error}",
            "code": "DB_ERROR"
        }

    # OpenAI/Azure specific errors
    if "openai" in error_str or "azure" in error_str:
        return {
            "message": "The AI service encountered an error. Please try again.",
            "details": f"AI service error: {error}",
            "code": "AI_ERROR"
        }

    # Default error
    return {
        "message": f"An error occurred while processing your {context}. Please try rephrasing your question or try again later.",
        "details": f"Unknown error ({error_type}): {error}",
        "code": "UNKNOWN_ERROR"
    }


def generate_answer(query: str, sources: list, session_id: str = None) -> dict:
    """
    Generate answer using Azure OpenAI with conversation memory.
    Returns dict with answer, doc_names, and any errors.
    """
    # Build context from sources with document names
    context_parts = []
    doc_names = []
    for i, source in enumerate(sources, 1):
        # Use smart truncation instead of hard cutoff
        text = smart_truncate(source["text"], max_chars=2500)
        # Create a friendly document name
        doc_name = source.get("source_file", "").split("/")[-1] if source.get("source_file") else source.get("portal", f"Document {i}")
        if doc_name.endswith(".pdf"):
            doc_name = doc_name[:-4].replace("_", " ").replace("-", " ").title()
        doc_names.append(doc_name)
        context_parts.append(f"[{doc_name}]\nSource: {source['portal']} - {source['section']}\n{text}\n")

    context = "\n".join(context_parts)
    num_sources = len(sources)

    # Get conversation history if session_id provided
    conversation_context = ""
    if session_id:
        history = get_conversation_history(session_id)
        conversation_context = format_conversation_context(history)

    system_prompt = f"""You are an expert assistant for the Uttar Pradesh Investment ecosystem. You help users understand:
- Investment policies and incentives in UP
- Government orders (Shasanadesh) related to industries
- Procedures for setting up businesses in UP
- Sector-specific guidelines and benefits
- Nivesh Mitra and other investment facilitation services

Answer based ONLY on the provided documents. If the information is not in the documents, say so.
You have access to {num_sources} documents. When citing, use the document name in brackets, e.g., [{doc_names[0] if doc_names else 'Document 1'}].
NEVER cite a document number higher than {num_sources}.
Be helpful, accurate, and concise.

IMPORTANT: ALWAYS respond in English, even if the user asks in Hindi or any other language.
You can understand queries in Hindi (or other languages), but your response MUST be in English.

IMPORTANT: If this is a follow-up question (like "tell me more", "what about X", "explain that"),
use the conversation history to understand the context and provide a relevant answer.

IMPORTANT FORMATTING RULES:
- Do NOT use markdown headers (no #, ##, ###, ####)
- Use **bold text** for emphasis and section titles
- Use bullet points (-) for lists
- Write in a clear, conversational tone
- Keep paragraphs short and readable"""

    user_prompt = f"""{conversation_context}Based on the following documents, answer this question:

Question: {query}

Documents:
{context}

Provide a comprehensive answer citing the relevant documents by name."""

    try:
        response = llm_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        answer = response.choices[0].message.content

        # Verify and fix citations
        answer = verify_citations(answer, num_sources, doc_names)

        # Check grounding
        grounding = check_answer_grounding(answer, sources)

        # Store in conversation history
        if session_id:
            add_to_conversation(session_id, query, answer)

        return {
            "answer": answer,
            "doc_names": doc_names,
            "grounding": grounding,
            "error": None
        }
    except Exception as e:
        error_info = get_specific_error_message(e, "question")
        print(f"[Generation] {error_info['details']}")
        return {
            "answer": None,
            "doc_names": doc_names,
            "grounding": None,
            "error": error_info
        }


@app.route("/")
def home():
    """Home page"""
    # Get index stats
    try:
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
    except:
        total_vectors = 0

    return render_template("index.html", stats={
        "total_documents": total_vectors,
        "portals": 6,
        "languages": 2
    })


@app.route("/api/search", methods=["POST"])
def api_search():
    """Search API endpoint with conversation memory, citation verification, and grounding check"""
    data = request.get_json()
    query = data.get("query", "").strip()

    # Get or create session ID for conversation memory
    session_id = data.get("session_id") or session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id

    if not query:
        return jsonify({
            "error": "Query is required",
            "error_code": "EMPTY_QUERY",
            "suggestion": "Please enter a question about UP investment policies, incentives, or procedures."
        }), 400

    # Validate query length
    if len(query) > 1000:
        return jsonify({
            "error": "Query is too long",
            "error_code": "QUERY_TOO_LONG",
            "suggestion": "Please limit your question to under 1000 characters."
        }), 400

    try:
        # Search for relevant documents using enhanced retrieval
        sources = search_documents(query)

        if not sources:
            return jsonify({
                "answer": "I couldn't find any relevant documents for your query. This could mean:\n\n- The topic might not be covered in our document database\n- Try using different keywords (e.g., 'subsidy' instead of 'grant')\n- Try asking about specific policies like 'Solar Energy Policy' or 'Startup Policy'",
                "sources": [],
                "num_sources": 0,
                "session_id": session_id,
                "grounding": None,
                "warning": None
            })

        # Add document names and links to sources for frontend
        for i, source in enumerate(sources):
            source_file = source.get("source_file", "")
            source_url = source.get("source_url", "")

            if source_file:
                filename = source_file.split("/")[-1]
                doc_name = filename
                if filename.endswith(".pdf"):
                    doc_name = filename[:-4].replace("_", " ").replace("-", " ").title()
                    # Create link to serve the PDF
                    source["link_url"] = f"/pdf/{filename}"
                else:
                    source["link_url"] = source_url if source_url else ""
            else:
                doc_name = source.get("portal", f"Document {i+1}")
                source["link_url"] = source_url if source_url else ""

            # If there's a source_url, prefer that for web content
            if source_url and not source_file:
                source["link_url"] = source_url

            source["doc_name"] = doc_name

        # Generate answer with conversation memory
        result = generate_answer(query, sources, session_id)

        # Handle generation errors
        if result["error"]:
            return jsonify({
                "error": result["error"]["message"],
                "error_code": result["error"]["code"],
                "sources": sources,
                "num_sources": len(sources),
                "session_id": session_id
            }), 500

        # Build response
        response_data = {
            "answer": result["answer"],
            "sources": sources,
            "num_sources": len(sources),
            "session_id": session_id,
            "grounding_score": result["grounding"]["score"] if result["grounding"] else None
        }

        # Add grounding warning if needed
        if result["grounding"] and result["grounding"]["warning"]:
            response_data["warning"] = result["grounding"]["warning"]

        return jsonify(response_data)

    except Exception as e:
        error_info = get_specific_error_message(e, "search")
        print(f"[API] {error_info['details']}")
        return jsonify({
            "error": error_info["message"],
            "error_code": error_info["code"]
        }), 500


@app.route("/api/stats")
def api_stats():
    """Get index statistics"""
    try:
        stats = index.describe_index_stats()
        return jsonify({
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/pdf/<path:filename>")
def serve_pdf(filename):
    """Serve PDF files from data directory"""
    # Security: only allow PDF files from the data/raw/pdf directory
    base_dir = Path(__file__).parent.parent / "data" / "raw" / "pdf"
    file_path = base_dir / filename

    # Ensure the file is within the allowed directory
    try:
        file_path = file_path.resolve()
        base_dir = base_dir.resolve()
        if not str(file_path).startswith(str(base_dir)):
            return jsonify({"error": "Access denied"}), 403
    except Exception:
        return jsonify({"error": "Invalid path"}), 400

    if file_path.exists() and file_path.suffix.lower() == '.pdf':
        return send_file(file_path, mimetype='application/pdf')
    else:
        return jsonify({"error": "File not found"}), 404


if __name__ == "__main__":
    app.run(debug=True, port=3006)
