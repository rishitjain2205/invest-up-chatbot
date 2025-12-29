"""
Enhanced Retrieval System for Invest UP Chatbot
- Hybrid Search: Vector (Pinecone) + BM25
- Cross-encoder Re-ranking
- Dynamic top_k adjustment
- Query processing (expansion, Hindi-English, spelling correction)
- LLM-based query rewriting for complex queries
"""
import json
import re
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Add parent for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CHUNKS_DIR

# Import query processor
from query_processor import get_query_processor

# LLM client for query rewriting (lazy loaded)
_llm_client = None

def get_llm_client():
    """Get LLM client for query rewriting (lazy loaded)"""
    global _llm_client
    if _llm_client is None:
        try:
            from config import (
                AZURE_OPENAI_ENDPOINT,
                AZURE_OPENAI_API_KEY,
                AZURE_OPENAI_API_VERSION,
            )
            from openai import AzureOpenAI
            _llm_client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
        except Exception as e:
            print(f"[QueryRewriter] Could not initialize LLM client: {e}")
    return _llm_client


class QueryRewriter:
    """
    LLM-based query rewriting for better retrieval.
    Handles:
    - Multi-part questions -> multiple focused queries
    - Vague questions -> specific queries
    - Complex questions -> simpler retrieval-friendly versions
    """

    REWRITE_PROMPT = """You are a query rewriter for a RAG system about Uttar Pradesh (UP) investment policies and business incentives.

Your task: Rewrite the user's query into 1-3 simpler, more specific search queries that will help find relevant documents.

Rules:
1. If the query is already simple and specific, return it as-is
2. If the query has multiple parts, split into separate queries
3. Focus on key terms: policy names, incentive types, sectors, procedures
4. Remove conversational filler words
5. Keep Hindi terms if present (they may match documents)
6. Return ONLY the rewritten queries, one per line, no numbering or explanation

Examples:
User: "I want to know what benefits I can get if I start a textile manufacturing unit in UP and also what is the process to apply"
Rewritten:
textile manufacturing incentives UP
textile unit subsidy benefits Uttar Pradesh
textile business application process nivesh mitra

User: "solar panel ke liye kya incentive milega"
Rewritten:
solar panel incentive UP
solar energy subsidy policy

User: "What are the benefits for MSME?"
Rewritten:
MSME benefits incentives UP policy

Now rewrite this query:
User: "{query}"
Rewritten:"""

    def __init__(self, deployment_name: str = None):
        self.deployment = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        self._cache = {}  # Simple cache for repeated queries

    def should_rewrite(self, query: str) -> bool:
        """Determine if query needs rewriting"""
        query_lower = query.lower()
        word_count = len(query.split())

        # Skip very short queries
        if word_count <= 3:
            return False

        # Rewrite if query has multiple questions
        if query.count('?') > 1:
            return True

        # Rewrite if query has conjunctions suggesting multiple parts
        multi_part_indicators = [' and ', ' also ', ' as well as ', ' along with ', ' plus ']
        if any(ind in query_lower for ind in multi_part_indicators):
            return True

        # Rewrite long conversational queries
        if word_count > 12:
            return True

        # Rewrite if contains vague terms
        vague_terms = ['want to know', 'tell me about', 'can you explain', 'what all', 'everything about']
        if any(term in query_lower for term in vague_terms):
            return True

        return False

    def rewrite(self, query: str) -> List[str]:
        """Rewrite query into retrieval-friendly versions"""
        # Check cache
        if query in self._cache:
            return self._cache[query]

        # If doesn't need rewriting, return original
        if not self.should_rewrite(query):
            return [query]

        try:
            client = get_llm_client()
            if client is None:
                return [query]

            response = client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "user", "content": self.REWRITE_PROMPT.format(query=query)}
                ],
                temperature=0.3,
                max_tokens=200
            )

            rewritten_text = response.choices[0].message.content.strip()

            # Parse response - one query per line
            queries = [q.strip() for q in rewritten_text.split('\n') if q.strip()]

            # Filter out any that look like explanations
            queries = [q for q in queries if not q.startswith(('Note:', 'Explanation:', '-', '*', '1.', '2.'))]

            # Limit to 3 queries max
            queries = queries[:3]

            if not queries:
                queries = [query]

            # Cache result
            self._cache[query] = queries

            print(f"[QueryRewriter] Original: '{query[:50]}...'")
            print(f"[QueryRewriter] Rewritten: {queries}")

            return queries

        except Exception as e:
            print(f"[QueryRewriter] Error: {e}, using original query")
            return [query]


# Global query rewriter instance
_query_rewriter = None

def get_query_rewriter() -> QueryRewriter:
    """Get or create query rewriter singleton"""
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter()
    return _query_rewriter


@dataclass
class RetrievalResult:
    """A single retrieval result with scores"""
    chunk_id: str
    text: str
    source_url: str
    source_file: str
    portal: str
    section: str
    doc_type: str
    chunk_index: int
    total_chunks: int
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


class HybridRetriever:
    """
    Hybrid retrieval combining:
    1. Dense vector search (Pinecone)
    2. Sparse BM25 search
    3. Cross-encoder re-ranking
    """

    def __init__(self, pinecone_index, embedding_func):
        self.index = pinecone_index
        self.get_embedding = embedding_func

        # Load chunks for BM25
        self.chunks = []
        self.chunk_texts = []
        self.chunk_id_to_idx = {}
        self.bm25 = None

        # Cross-encoder for re-ranking (lightweight model)
        self.cross_encoder = None
        self._load_cross_encoder()

        # Load BM25 index
        self._load_bm25_index()

    def _load_cross_encoder(self):
        """Load cross-encoder model for re-ranking"""
        try:
            # Use a small, fast cross-encoder
            self.cross_encoder = CrossEncoder(
                'cross-encoder/ms-marco-MiniLM-L-6-v2',
                max_length=512
            )
            print("[Retriever] Cross-encoder loaded successfully")
        except Exception as e:
            print(f"[Retriever] Warning: Could not load cross-encoder: {e}")
            self.cross_encoder = None

    def _load_bm25_index(self):
        """Load chunks and build BM25 index"""
        chunks_file = CHUNKS_DIR / "all_chunks.json"

        if not chunks_file.exists():
            print("[Retriever] Warning: No chunks file found for BM25")
            return

        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)

            # Build tokenized corpus for BM25
            tokenized_corpus = []
            for i, chunk in enumerate(self.chunks):
                text = chunk.get("text", "")
                # Simple tokenization: lowercase, split on non-alphanumeric
                tokens = self._tokenize(text)
                tokenized_corpus.append(tokens)
                self.chunk_texts.append(text)
                self.chunk_id_to_idx[chunk.get("chunk_id", str(i))] = i

            # Build BM25 index
            if tokenized_corpus:
                self.bm25 = BM25Okapi(tokenized_corpus)
                print(f"[Retriever] BM25 index built with {len(self.chunks)} chunks")

        except Exception as e:
            print(f"[Retriever] Error loading BM25 index: {e}")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        # Lowercase and split on non-alphanumeric (preserves Hindi)
        text = text.lower()
        tokens = re.findall(r'[\w]+', text, re.UNICODE)
        return tokens

    def calculate_dynamic_top_k(self, query: str) -> Tuple[int, int]:
        """
        Calculate dynamic top_k based on query characteristics
        Returns: (initial_k for retrieval, final_k after re-ranking)
        """
        query_lower = query.lower()
        word_count = len(query.split())

        # Short queries (1-3 words) - need more candidates
        if word_count <= 3:
            initial_k = 15
            final_k = 7
        # Medium queries (4-8 words) - moderate candidates
        elif word_count <= 8:
            initial_k = 12
            final_k = 6
        # Long/specific queries (9+ words) - fewer but targeted
        else:
            initial_k = 10
            final_k = 5

        # Boost for question words (more exploratory)
        question_words = ['what', 'how', 'why', 'which', 'explain', 'describe', 'list']
        if any(qw in query_lower for qw in question_words):
            initial_k += 3
            final_k += 1

        # Reduce for very specific queries (has numbers, specific terms)
        if re.search(r'\d{4,}', query):  # Year or order numbers
            initial_k = max(8, initial_k - 3)
            final_k = max(4, final_k - 1)

        return initial_k, final_k

    def search_vector(self, query: str, top_k: int) -> List[Dict]:
        """Search using vector similarity (Pinecone)"""
        try:
            query_embedding = self.get_embedding(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )

            return [
                {
                    "chunk_id": match.id,
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
        except Exception as e:
            print(f"[Retriever] Vector search error: {e}")
            return []

    def search_bm25(self, query: str, top_k: int) -> List[Dict]:
        """Search using BM25"""
        if self.bm25 is None:
            return []

        try:
            query_tokens = self._tokenize(query)
            scores = self.bm25.get_scores(query_tokens)

            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include if there's some match
                    chunk = self.chunks[idx]
                    results.append({
                        "chunk_id": chunk.get("chunk_id", str(idx)),
                        "score": float(scores[idx]),
                        "text": chunk.get("text", ""),
                        "source_url": chunk.get("source_url", ""),
                        "source_file": chunk.get("source_file", ""),
                        "portal": chunk.get("portal", ""),
                        "section": chunk.get("section", ""),
                        "doc_type": chunk.get("doc_type", ""),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "total_chunks": chunk.get("total_chunks", 1)
                    })

            return results
        except Exception as e:
            print(f"[Retriever] BM25 search error: {e}")
            return []

    def hybrid_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        Combine vector and BM25 search results using Reciprocal Rank Fusion (RRF)
        """
        # Get results from both systems
        vector_results = self.search_vector(query, top_k)
        bm25_results = self.search_bm25(query, top_k)

        # Create unified result dict
        results_dict: Dict[str, RetrievalResult] = {}

        # Add vector results
        for rank, result in enumerate(vector_results):
            chunk_id = result["chunk_id"]
            if chunk_id not in results_dict:
                results_dict[chunk_id] = RetrievalResult(
                    chunk_id=chunk_id,
                    text=result["text"],
                    source_url=result["source_url"],
                    source_file=result["source_file"],
                    portal=result["portal"],
                    section=result["section"],
                    doc_type=result["doc_type"],
                    chunk_index=result["chunk_index"],
                    total_chunks=result["total_chunks"]
                )
            results_dict[chunk_id].vector_score = result["score"]

        # Add BM25 results
        for rank, result in enumerate(bm25_results):
            chunk_id = result["chunk_id"]
            if chunk_id not in results_dict:
                results_dict[chunk_id] = RetrievalResult(
                    chunk_id=chunk_id,
                    text=result["text"],
                    source_url=result["source_url"],
                    source_file=result["source_file"],
                    portal=result["portal"],
                    section=result["section"],
                    doc_type=result["doc_type"],
                    chunk_index=result["chunk_index"],
                    total_chunks=result["total_chunks"]
                )
            results_dict[chunk_id].bm25_score = result["score"]

        # Calculate RRF scores
        # RRF = sum(1 / (k + rank)) for each system
        k = 60  # RRF constant

        # Sort vector results by score to get ranks
        vector_ranked = sorted(
            [(r["chunk_id"], r["score"]) for r in vector_results],
            key=lambda x: x[1], reverse=True
        )
        vector_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(vector_ranked)}

        # Sort BM25 results by score to get ranks
        bm25_ranked = sorted(
            [(r["chunk_id"], r["score"]) for r in bm25_results],
            key=lambda x: x[1], reverse=True
        )
        bm25_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(bm25_ranked)}

        # Calculate final RRF score
        for chunk_id, result in results_dict.items():
            rrf_score = 0.0
            if chunk_id in vector_ranks:
                rrf_score += 1.0 / (k + vector_ranks[chunk_id])
            if chunk_id in bm25_ranks:
                rrf_score += 1.0 / (k + bm25_ranks[chunk_id])
            result.final_score = rrf_score

        # Sort by final score
        sorted_results = sorted(
            results_dict.values(),
            key=lambda x: x.final_score,
            reverse=True
        )

        return sorted_results[:top_k]

    def hybrid_search_enhanced(self, vector_query: str, bm25_query: str, top_k: int) -> List[RetrievalResult]:
        """
        Enhanced hybrid search with separate queries for vector and BM25.
        - vector_query: Processed query for semantic vector search
        - bm25_query: Expanded query with synonyms for keyword matching
        """
        # Get results from both systems with their respective queries
        vector_results = self.search_vector(vector_query, top_k)
        bm25_results = self.search_bm25(bm25_query, top_k)

        # Create unified result dict
        results_dict: Dict[str, RetrievalResult] = {}

        # Add vector results
        for rank, result in enumerate(vector_results):
            chunk_id = result["chunk_id"]
            if chunk_id not in results_dict:
                results_dict[chunk_id] = RetrievalResult(
                    chunk_id=chunk_id,
                    text=result["text"],
                    source_url=result["source_url"],
                    source_file=result["source_file"],
                    portal=result["portal"],
                    section=result["section"],
                    doc_type=result["doc_type"],
                    chunk_index=result["chunk_index"],
                    total_chunks=result["total_chunks"]
                )
            results_dict[chunk_id].vector_score = result["score"]

        # Add BM25 results
        for rank, result in enumerate(bm25_results):
            chunk_id = result["chunk_id"]
            if chunk_id not in results_dict:
                results_dict[chunk_id] = RetrievalResult(
                    chunk_id=chunk_id,
                    text=result["text"],
                    source_url=result["source_url"],
                    source_file=result["source_file"],
                    portal=result["portal"],
                    section=result["section"],
                    doc_type=result["doc_type"],
                    chunk_index=result["chunk_index"],
                    total_chunks=result["total_chunks"]
                )
            results_dict[chunk_id].bm25_score = result["score"]

        # Calculate RRF scores
        k = 60  # RRF constant

        # Sort vector results by score to get ranks
        vector_ranked = sorted(
            [(r["chunk_id"], r["score"]) for r in vector_results],
            key=lambda x: x[1], reverse=True
        )
        vector_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(vector_ranked)}

        # Sort BM25 results by score to get ranks
        bm25_ranked = sorted(
            [(r["chunk_id"], r["score"]) for r in bm25_results],
            key=lambda x: x[1], reverse=True
        )
        bm25_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(bm25_ranked)}

        # Calculate final RRF score
        for chunk_id, result in results_dict.items():
            rrf_score = 0.0
            if chunk_id in vector_ranks:
                rrf_score += 1.0 / (k + vector_ranks[chunk_id])
            if chunk_id in bm25_ranks:
                rrf_score += 1.0 / (k + bm25_ranks[chunk_id])
            result.final_score = rrf_score

        # Sort by final score
        sorted_results = sorted(
            results_dict.values(),
            key=lambda x: x.final_score,
            reverse=True
        )

        return sorted_results[:top_k]

    def rerank(self, query: str, results: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        """Re-rank results using cross-encoder"""
        if self.cross_encoder is None or len(results) == 0:
            return results[:top_k]

        try:
            # Prepare query-document pairs
            pairs = [(query, r.text[:500]) for r in results]  # Limit text length

            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)

            # Update results with rerank scores
            for i, result in enumerate(results):
                result.rerank_score = float(scores[i])

            # Sort by rerank score
            reranked = sorted(results, key=lambda x: x.rerank_score, reverse=True)

            return reranked[:top_k]
        except Exception as e:
            print(f"[Retriever] Re-ranking error: {e}")
            return results[:top_k]

    def search(self, query: str) -> List[Dict]:
        """
        Main search method combining all techniques:
        1. Query rewriting (for complex queries)
        2. Query processing (spelling, Hindi-English, expansion)
        3. Dynamic top_k calculation
        4. Multi-query hybrid search (vector + BM25)
        5. Cross-encoder re-ranking
        """
        # Step 1: Query rewriting for complex queries
        try:
            rewriter = get_query_rewriter()
            rewritten_queries = rewriter.rewrite(query)
        except Exception as e:
            print(f"[QueryRewriter] Error: {e}, using original query")
            rewritten_queries = [query]

        # Collect all results from all query variants
        all_results: Dict[str, RetrievalResult] = {}

        for q_idx, rewritten_q in enumerate(rewritten_queries):
            # Step 2: Process each query (spelling correction, Hindi translation, expansion)
            try:
                qp = get_query_processor()
                processed = qp.process(rewritten_q)
                search_query = processed["search_query"]
                expanded_query = processed["expanded"]

                if processed["modifications"] and q_idx == 0:  # Only log first query
                    print(f"[QueryProcessor] Original: '{rewritten_q[:40]}...'")
                    print(f"[QueryProcessor] Processed: '{search_query[:40]}...'")
            except Exception as e:
                print(f"[QueryProcessor] Error: {e}, using original query")
                search_query = rewritten_q
                expanded_query = rewritten_q

            # Step 3: Calculate dynamic k values
            initial_k, final_k = self.calculate_dynamic_top_k(search_query)

            # Adjust k for multi-query - fetch more per query, dedupe later
            if len(rewritten_queries) > 1:
                initial_k = max(8, initial_k // len(rewritten_queries) + 3)

            print(f"[Retriever] Query {q_idx+1}/{len(rewritten_queries)}: '{search_query[:40]}...' | k={initial_k}")

            # Step 4: Hybrid search - vector + BM25
            hybrid_results = self.hybrid_search_enhanced(search_query, expanded_query, initial_k)

            # Merge results, boosting those that appear in multiple queries
            for result in hybrid_results:
                if result.chunk_id in all_results:
                    # Boost score for results appearing in multiple query variants
                    existing = all_results[result.chunk_id]
                    existing.final_score += result.final_score * 0.5  # Diminishing boost
                    existing.vector_score = max(existing.vector_score, result.vector_score)
                    existing.bm25_score = max(existing.bm25_score, result.bm25_score)
                else:
                    all_results[result.chunk_id] = result

        # Calculate final k based on original query
        _, final_k = self.calculate_dynamic_top_k(query)

        # Sort merged results by final score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.final_score,
            reverse=True
        )

        print(f"[Retriever] Multi-query search returned {len(sorted_results)} unique results")

        # Step 5: Re-rank using original query for better semantic matching
        # Use original query for reranking (not rewritten) for semantic coherence
        reranked_results = self.rerank(query, sorted_results[:final_k * 2], final_k)

        print(f"[Retriever] After re-ranking: {len(reranked_results)} results")

        # Convert to dict format for compatibility
        return [
            {
                "id": r.chunk_id,
                "score": r.rerank_score if r.rerank_score > 0 else r.final_score,
                "text": r.text,
                "source_url": r.source_url,
                "source_file": r.source_file,
                "portal": r.portal,
                "section": r.section,
                "doc_type": r.doc_type,
                "chunk_index": r.chunk_index,
                "total_chunks": r.total_chunks,
                # Debug scores
                "_vector_score": r.vector_score,
                "_bm25_score": r.bm25_score,
                "_rrf_score": r.final_score,
                "_rerank_score": r.rerank_score
            }
            for r in reranked_results
        ]


# Singleton instance
_retriever_instance: Optional[HybridRetriever] = None


def get_retriever(pinecone_index, embedding_func) -> HybridRetriever:
    """Get or create retriever singleton"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever(pinecone_index, embedding_func)
    return _retriever_instance
