"""
RAG Retriever with Azure OpenAI for generation
"""
from typing import List, Dict, Optional
from dataclasses import dataclass

from openai import AzureOpenAI
from rich.console import Console

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    EMBEDDING_DIMENSIONS,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS
)
from rag.vector_store import PineconeVectorStore
from rag.embeddings import EmbeddingGenerator

console = Console()


@dataclass
class RAGResponse:
    """Response from RAG system"""
    answer: str
    sources: List[Dict]
    query: str


class InvestUPRetriever:
    """RAG system for Invest UP chatbot"""

    SYSTEM_PROMPT = """You are an expert assistant for the Invest UP portal (invest.up.gov.in), the official investment promotion agency of Uttar Pradesh, India.

Your role is to help users understand:
- Investment policies and incentives in Uttar Pradesh
- Government orders (GOs) and circulars
- Sector-specific opportunities (IT, manufacturing, agriculture, etc.)
- Procedures for setting up industries
- Available schemes and benefits for investors

Guidelines:
1. ONLY answer based on the provided context from official documents
2. If the information is not in the context, say "I don't have specific information about that in the available documents"
3. Always cite your sources with document names or URLs when possible
4. Be precise with numbers, percentages, and incentive details
5. If a question is in Hindi, understand it but respond in English
6. For policy-related questions, mention the policy name and year
7. For procedural questions, provide step-by-step guidance if available

Format your responses clearly with:
- Direct answer first
- Supporting details
- Source references at the end"""

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        self.vector_store = PineconeVectorStore()
        self.embedding_generator = EmbeddingGenerator()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        section_filter: Optional[str] = None
    ) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)

        if not query_embedding:
            console.print("[red]Failed to generate query embedding[/red]")
            return []

        # Build filter if section specified
        filter_dict = None
        if section_filter:
            filter_dict = {"section": {"$eq": section_filter}}

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )

        return results

    def generate_answer(
        self,
        query: str,
        context_docs: List[Dict]
    ) -> str:
        """Generate answer using Azure OpenAI"""
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source = doc.get("source_url") or doc.get("source_file", "Unknown")
            section = doc.get("section", "general")
            text = doc.get("text", "")

            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {source}\n"
                f"Section: {section}\n"
                f"Content:\n{text}\n"
            )

        context = "\n---\n".join(context_parts)

        # Build messages
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Based on the following documents from the Invest UP portal, please answer the user's question.

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {query}

Please provide a comprehensive answer based on the above documents."""
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            console.print(f"[red]Error generating answer: {e}[/red]")
            return f"I encountered an error while generating the response: {str(e)}"

    def query(
        self,
        question: str,
        top_k: int = 5,
        section_filter: Optional[str] = None
    ) -> RAGResponse:
        """Full RAG pipeline: retrieve + generate"""
        console.print(f"[blue]Query: {question}[/blue]")

        # Retrieve relevant documents
        docs = self.retrieve(question, top_k=top_k, section_filter=section_filter)

        if not docs:
            return RAGResponse(
                answer="I couldn't find any relevant documents to answer your question. Please try rephrasing or ask about a different topic.",
                sources=[],
                query=question
            )

        console.print(f"[green]Retrieved {len(docs)} relevant documents[/green]")

        # Generate answer
        answer = self.generate_answer(question, docs)

        # Prepare sources
        sources = [
            {
                "url": doc.get("source_url", ""),
                "file": doc.get("source_file", ""),
                "section": doc.get("section", ""),
                "score": doc.get("score", 0)
            }
            for doc in docs
        ]

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question
        )


def main():
    """Interactive testing"""
    retriever = InvestUPRetriever()

    console.print("[bold green]Invest UP Chatbot Ready![/bold green]")
    console.print("Type 'quit' to exit\n")

    while True:
        try:
            question = input("\nYour question: ").strip()
            if question.lower() == 'quit':
                break

            response = retriever.query(question)

            console.print("\n[bold]Answer:[/bold]")
            console.print(response.answer)

            console.print("\n[bold]Sources:[/bold]")
            for source in response.sources:
                if source["url"]:
                    console.print(f"  - {source['url']} (score: {source['score']:.3f})")
                elif source["file"]:
                    console.print(f"  - {source['file']} (score: {source['score']:.3f})")

        except KeyboardInterrupt:
            break

    console.print("\n[yellow]Goodbye![/yellow]")


if __name__ == "__main__":
    main()
