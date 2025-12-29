"""
Enhanced Document Chunking for Invest UP RAG Pipeline
- Semantic boundary-aware splitting (headers, sections, lists)
- Table detection and preservation
- Hindi/Devanagari text handling
- Smart overlap at sentence/paragraph boundaries
"""
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import tiktoken
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, CHUNKS_DIR, LOGS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

console = Console()


@dataclass
class DocumentChunk:
    """A chunk of document text for embedding"""
    chunk_id: str
    text: str
    source_file: str
    source_url: str
    portal: str
    doc_type: str
    section: str
    chunk_index: int
    total_chunks: int
    token_count: int
    metadata: Dict


class SemanticChunker:
    """
    Enhanced chunker with:
    1. Semantic boundary detection (headers, tables, lists)
    2. Table preservation (keeps tables intact)
    3. Hindi-aware processing
    4. Smart overlap at natural boundaries
    """

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunks: List[DocumentChunk] = []

        # Regex patterns for semantic boundaries
        self.table_pattern = re.compile(
            r'(\|[^\n]+\|[\r\n]+)+',  # Markdown tables
            re.MULTILINE
        )
        self.table_pattern_alt = re.compile(
            r'(?:[\t ]*[^\t\n]+[\t ]+[^\t\n]+[\t ]*\n){2,}',  # Tab/space separated tables
            re.MULTILINE
        )
        # Header patterns (markdown and plain text)
        self.header_pattern = re.compile(
            r'^(?:#{1,6}\s+.+|[A-Z][A-Z\s]{5,}:?|(?:\d+\.)+\s+[A-Z].+)$',
            re.MULTILINE
        )
        # List patterns
        self.list_pattern = re.compile(
            r'^[\s]*[-*\u2022\u25cf\u25aa]\s+.+$|^[\s]*\d+[.)]\s+.+$',
            re.MULTILINE
        )
        # Hindi text pattern (Devanagari script)
        self.hindi_pattern = re.compile(r'[\u0900-\u097F]')

        # Sentence endings for multiple languages
        self.sentence_end_pattern = re.compile(
            r'(?<=[.!?\u0964\u0965])\s+',  # Includes Hindi danda and double danda
            re.UNICODE
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))

    def is_hindi_heavy(self, text: str) -> bool:
        """Check if text is predominantly Hindi"""
        hindi_chars = len(self.hindi_pattern.findall(text))
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        if total_chars == 0:
            return False
        return hindi_chars / total_chars > 0.3

    def generate_chunk_id(self, source: str, index: int) -> str:
        """Generate unique chunk ID"""
        hash_input = f"{source}:{index}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def create_document_context(self, doc_title: str, portal: str, section: str,
                                 chunk_index: int, total_chunks: int, source_file: str = "") -> str:
        """
        Create a document context header to prepend to each chunk.
        This helps the LLM understand the source and position of the content.
        """
        # Clean up document title
        if not doc_title and source_file:
            # Extract title from filename
            filename = Path(source_file).stem
            doc_title = filename.replace("_", " ").replace("-", " ").title()
            # Clean up common patterns
            doc_title = re.sub(r'^Go\s+\d+\s+', 'Government Order ', doc_title)

        # Format section nicely
        section_display = section.replace("_", " ").title() if section else "General"

        # Create context header
        context_parts = []

        if doc_title:
            context_parts.append(f"Document: {doc_title}")
        if portal and portal != "unknown":
            context_parts.append(f"Source: {portal}")
        if section_display:
            context_parts.append(f"Section: {section_display}")
        if total_chunks > 1:
            context_parts.append(f"[Part {chunk_index + 1} of {total_chunks}]")

        if context_parts:
            return "---\n" + " | ".join(context_parts) + "\n---\n\n"
        return ""

    def detect_section(self, source_url: str, source_file: str, portal: str) -> str:
        """Detect document section from URL, filename, or portal"""
        url_lower = (source_url or "").lower()
        file_lower = (source_file or "").lower()

        if "go" in file_lower or "/gos/" in url_lower or "government-order" in url_lower:
            return "government_order"
        elif "circular" in file_lower or "/circular" in url_lower:
            return "circular"
        elif "policy" in file_lower or "/polic" in url_lower:
            return "policy"
        elif "startup" in url_lower or "startinup" in portal:
            return "startup"
        elif "incubator" in url_lower or "coe" in url_lower:
            return "incubator"
        elif "expressway" in url_lower or "upeida" in portal:
            return "infrastructure"
        elif "upsida" in portal.lower() or "industrial-area" in url_lower:
            return "industrial_area"
        elif "sector" in url_lower or any(s in url_lower for s in [
            "it-ites", "textile", "pharma", "agriculture", "tourism",
            "renewable", "ev-", "semiconductor", "defence", "msme"
        ]):
            return "sector"
        elif "/faq" in url_lower:
            return "faq"
        elif "scheme" in file_lower or "incentive" in file_lower:
            return "scheme"
        elif "newsletter" in url_lower:
            return "newsletter"
        elif "tender" in url_lower:
            return "tender"
        elif "manual" in file_lower or "sop" in file_lower:
            return "procedure"
        elif "niveshmitra" in portal:
            return "procedure"
        else:
            return "general"

    def extract_tables(self, text: str) -> Tuple[str, List[Tuple[int, str]]]:
        """
        Extract tables from text to preserve them as atomic units.
        Returns: (text_without_tables, [(position, table_text), ...])
        """
        tables = []

        # Find markdown tables
        for match in self.table_pattern.finditer(text):
            tables.append((match.start(), match.group()))

        # Find tab-separated tables if no markdown tables found
        if not tables:
            for match in self.table_pattern_alt.finditer(text):
                table_text = match.group()
                # Only if it looks like a real table (multiple rows with consistent columns)
                rows = table_text.strip().split('\n')
                if len(rows) >= 2:
                    tables.append((match.start(), table_text))

        # Remove tables from text and replace with placeholders
        # Use double newlines to ensure placeholder becomes separate block
        modified_text = text
        for i, (pos, table) in enumerate(sorted(tables, key=lambda x: -x[0])):
            placeholder = f"\n\n[TABLE_{i}]\n\n"
            modified_text = modified_text[:pos] + placeholder + modified_text[pos + len(table):]

        return modified_text, tables

    def split_into_semantic_blocks(self, text: str) -> List[Dict]:
        """
        Split text into semantic blocks: headers, paragraphs, lists, tables.
        Each block has: type, content, token_count
        """
        blocks = []

        # Extract tables first
        text_no_tables, tables = self.extract_tables(text)
        table_dict = {f"[TABLE_{i}]": t[1] for i, t in enumerate(tables)}

        # Split by double newlines (paragraphs)
        raw_blocks = re.split(r'\n\n+', text_no_tables)

        for raw_block in raw_blocks:
            raw_block = raw_block.strip()
            if not raw_block:
                continue

            # Check for table placeholder
            if raw_block.startswith('[TABLE_') and raw_block.endswith(']'):
                table_content = table_dict.get(raw_block, "")
                if table_content:
                    blocks.append({
                        'type': 'table',
                        'content': table_content,
                        'tokens': self.count_tokens(table_content),
                        'preserve': True  # Don't split tables
                    })
                continue

            # Check if it's a header
            if self.header_pattern.match(raw_block):
                blocks.append({
                    'type': 'header',
                    'content': raw_block,
                    'tokens': self.count_tokens(raw_block),
                    'preserve': False
                })
                continue

            # Check if it's a list
            lines = raw_block.split('\n')
            list_lines = [l for l in lines if self.list_pattern.match(l)]
            if len(list_lines) > len(lines) * 0.5:  # Majority are list items
                blocks.append({
                    'type': 'list',
                    'content': raw_block,
                    'tokens': self.count_tokens(raw_block),
                    'preserve': True  # Try to keep lists together
                })
                continue

            # Regular paragraph
            blocks.append({
                'type': 'paragraph',
                'content': raw_block,
                'tokens': self.count_tokens(raw_block),
                'preserve': False
            })

        return blocks

    def split_long_block(self, block: Dict, max_tokens: int) -> List[str]:
        """
        Split a long block (paragraph) into smaller pieces at sentence boundaries.
        """
        text = block['content']
        chunks = []

        # Split by sentences (handles both English and Hindi endings)
        sentences = self.sentence_end_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)

            # If single sentence exceeds max, split by words
            if sent_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_tokens = 0

                # Split long sentence by words
                words = sentence.split()
                word_chunk = ""
                word_tokens = 0
                for word in words:
                    wt = self.count_tokens(word + " ")
                    if word_tokens + wt > max_tokens:
                        if word_chunk:
                            chunks.append(word_chunk.strip())
                        word_chunk = word
                        word_tokens = wt
                    else:
                        word_chunk += " " + word if word_chunk else word
                        word_tokens += wt
                if word_chunk:
                    chunks.append(word_chunk.strip())
                continue

            # Normal case: add sentence to current chunk
            if current_tokens + sent_tokens <= max_tokens:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sent_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
                current_tokens = sent_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def get_smart_overlap(self, text: str, target_tokens: int) -> str:
        """
        Get overlap text that ends at a sentence boundary.
        Returns the last N tokens worth of text, but ending at a sentence.
        """
        # Split into sentences
        sentences = self.sentence_end_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return ""

        # Build overlap from the end, sentence by sentence
        overlap_parts = []
        current_tokens = 0

        for sentence in reversed(sentences):
            sent_tokens = self.count_tokens(sentence)
            if current_tokens + sent_tokens > target_tokens * 1.2:  # Allow 20% overshoot
                break
            overlap_parts.insert(0, sentence)
            current_tokens += sent_tokens
            if current_tokens >= target_tokens * 0.8:  # At least 80% of target
                break

        return " ".join(overlap_parts) if overlap_parts else ""

    def split_text(self, text: str, max_tokens: int = CHUNK_SIZE) -> List[str]:
        """
        Main splitting method with semantic awareness:
        1. Split into semantic blocks (headers, paragraphs, lists, tables)
        2. Group blocks up to max_tokens
        3. Keep tables and lists intact when possible
        4. Add smart overlap at sentence boundaries
        """
        # Get semantic blocks
        blocks = self.split_into_semantic_blocks(text)

        if not blocks:
            return []

        chunks = []
        current_chunk_parts = []
        current_tokens = 0

        for block in blocks:
            block_tokens = block['tokens']

            # Handle oversized blocks
            if block_tokens > max_tokens:
                # First, save current chunk
                if current_chunk_parts:
                    chunks.append("\n\n".join(current_chunk_parts))
                    current_chunk_parts = []
                    current_tokens = 0

                # Split the large block
                if block['type'] == 'table':
                    # Tables: keep as single chunk even if large
                    chunks.append(block['content'])
                else:
                    # Split long paragraph at sentence boundaries
                    sub_chunks = self.split_long_block(block, max_tokens)
                    chunks.extend(sub_chunks)
                continue

            # Check if adding this block exceeds limit
            if current_tokens + block_tokens > max_tokens:
                # Save current chunk
                if current_chunk_parts:
                    chunks.append("\n\n".join(current_chunk_parts))

                # Start new chunk with this block
                current_chunk_parts = [block['content']]
                current_tokens = block_tokens
            else:
                # Add block to current chunk
                current_chunk_parts.append(block['content'])
                current_tokens += block_tokens

        # Don't forget the last chunk
        if current_chunk_parts:
            chunks.append("\n\n".join(current_chunk_parts))

        # Add smart overlap between chunks
        if CHUNK_OVERLAP > 0 and len(chunks) > 1:
            overlapped_chunks = [chunks[0]]

            for i in range(1, len(chunks)):
                prev_chunk = chunks[i - 1]
                current = chunks[i]

                # Get smart overlap from previous chunk
                overlap = self.get_smart_overlap(prev_chunk, CHUNK_OVERLAP)

                if overlap and not current.startswith(overlap[:50]):  # Avoid duplication
                    overlapped_chunks.append(f"[...] {overlap}\n\n{current}")
                else:
                    overlapped_chunks.append(current)

            return overlapped_chunks

        return chunks

    def chunk_parsed_documents(self) -> int:
        """Chunk all parsed PDF documents"""
        console.print("\n[bold blue]Chunking parsed documents...[/bold blue]")

        parsed_file = PROCESSED_DIR / "parsed_documents.json"
        if not parsed_file.exists():
            console.print("[yellow]No parsed documents found[/yellow]")
            return 0

        with open(parsed_file, "r", encoding="utf-8") as f:
            parsed_docs = json.load(f)

        initial_count = len(self.chunks)
        hindi_docs = 0
        table_preserved = 0

        with Progress() as progress:
            task = progress.add_task("Chunking PDFs...", total=len(parsed_docs))

            for file_path, doc_data in parsed_docs.items():
                text = doc_data.get("text", "")
                if not text or len(text) < 50:
                    progress.update(task, advance=1)
                    continue

                source_url = doc_data.get("original_url", "")
                portal = doc_data.get("portal", "unknown")
                section = self.detect_section(source_url, file_path, portal)

                # Track Hindi documents
                is_hindi = self.is_hindi_heavy(text)
                if is_hindi:
                    hindi_docs += 1

                # Check for tables
                has_tables = bool(self.table_pattern.search(text) or
                                self.table_pattern_alt.search(text))
                if has_tables:
                    table_preserved += 1

                # Get document title from metadata or filename
                doc_title = doc_data.get("metadata", {}).get("title", "")

                text_chunks = self.split_text(text)

                for i, chunk_text in enumerate(text_chunks):
                    # Add document context header to chunk
                    context_header = self.create_document_context(
                        doc_title=doc_title,
                        portal=portal,
                        section=section,
                        chunk_index=i,
                        total_chunks=len(text_chunks),
                        source_file=file_path
                    )
                    contextualized_text = context_header + chunk_text

                    chunk = DocumentChunk(
                        chunk_id=self.generate_chunk_id(file_path, i),
                        text=contextualized_text,
                        source_file=file_path,
                        source_url=source_url,
                        portal=portal,
                        doc_type=doc_data.get("doc_type", "pdf"),
                        section=section,
                        chunk_index=i,
                        total_chunks=len(text_chunks),
                        token_count=self.count_tokens(contextualized_text),
                        metadata={
                            "filename": Path(file_path).name,
                            "num_pages": doc_data.get("num_pages", 0),
                            "title": doc_title or Path(file_path).stem,
                            "has_hindi": is_hindi,
                            "has_tables": has_tables,
                            "has_context_header": True
                        }
                    )
                    self.chunks.append(chunk)

                progress.update(task, advance=1)

        new_chunks = len(self.chunks) - initial_count
        console.print(f"[green]Created {new_chunks} chunks from PDFs[/green]")
        console.print(f"  - Hindi-heavy documents: {hindi_docs}")
        console.print(f"  - Documents with tables: {table_preserved}")
        return new_chunks

    def chunk_html_content(self) -> int:
        """Chunk HTML content from crawl"""
        console.print("\n[bold blue]Chunking HTML content...[/bold blue]")

        content_file = LOGS_DIR / "all_portal_content.json"
        if not content_file.exists():
            content_file = PROCESSED_DIR / "html_content.json"

        if not content_file.exists():
            console.print("[yellow]No HTML content found[/yellow]")
            return 0

        try:
            console.print(f"[dim]Loading HTML content from {content_file}...[/dim]")
            with open(content_file, "r", encoding="utf-8") as f:
                html_content = json.load(f)
            console.print(f"[dim]Loaded {len(html_content)} HTML pages[/dim]")
        except Exception as e:
            console.print(f"[red]Error loading HTML content: {e}[/red]")
            return 0

        initial_count = len(self.chunks)

        # Use simple print-based progress instead of rich Progress for stability
        total = len(html_content)
        console.print(f"[dim]Processing {total} HTML pages...[/dim]")

        processed = 0
        for url, text in html_content.items():
            processed += 1
            if processed % 50 == 0:
                console.print(f"[dim]HTML progress: {processed}/{total}[/dim]")

            if not text or len(text) < 100:
                continue

            # Determine portal from URL
            portal = "unknown"
            if "invest.up.gov.in" in url:
                portal = "invest.up.gov.in"
            elif "niveshmitra" in url:
                portal = "niveshmitra.up.nic.in"
            elif "niveshsarathi" in url:
                portal = "niveshsarathi.up.gov.in"
            elif "startinup" in url:
                portal = "startinup.up.gov.in"
            elif "upeida" in url:
                portal = "upeida.up.gov.in"
            elif "upsida" in url.lower():
                portal = "UPSIDA"

            section = self.detect_section(url, "", portal)
            text_chunks = self.split_text(text)

            # Extract page title from URL
            page_title = url.split("/")[-1].replace("-", " ").replace("_", " ").title()
            if page_title.endswith(".Html") or page_title.endswith(".Php"):
                page_title = page_title.rsplit(".", 1)[0]

            for i, chunk_text in enumerate(text_chunks):
                # Add document context header
                context_header = self.create_document_context(
                    doc_title=page_title,
                    portal=portal,
                    section=section,
                    chunk_index=i,
                    total_chunks=len(text_chunks)
                )
                contextualized_text = context_header + chunk_text

                chunk = DocumentChunk(
                    chunk_id=self.generate_chunk_id(url, i),
                    text=contextualized_text,
                    source_file="",
                    source_url=url,
                    portal=portal,
                    doc_type="html",
                    section=section,
                    chunk_index=i,
                    total_chunks=len(text_chunks),
                    token_count=self.count_tokens(contextualized_text),
                    metadata={
                        "page_url": url,
                        "has_hindi": self.is_hindi_heavy(chunk_text),
                        "has_context_header": True
                    }
                )
                self.chunks.append(chunk)

        new_chunks = len(self.chunks) - initial_count
        console.print(f"\n[green]Created {new_chunks} chunks from HTML[/green]")
        return new_chunks

    def chunk_all(self) -> List[DocumentChunk]:
        """Chunk all content (PDFs + HTML)"""
        console.print("\n[bold blue]========================================[/bold blue]")
        console.print("[bold blue]  Semantic Document Chunking[/bold blue]")
        console.print("[bold blue]========================================[/bold blue]")

        pdf_chunks = self.chunk_parsed_documents()

        # Save checkpoint after PDF chunking (in case HTML hangs)
        if pdf_chunks > 0:
            console.print(f"\n[yellow]Saving PDF chunks checkpoint ({pdf_chunks} chunks)...[/yellow]")
            self.save_chunks()

        html_chunks = self.chunk_html_content()

        # Save all chunks
        self.save_chunks()

        # Print summary
        console.print(f"\n[bold green]Chunking Summary:[/bold green]")
        console.print(f"  PDF chunks: {pdf_chunks}")
        console.print(f"  HTML chunks: {html_chunks}")
        console.print(f"  Total chunks: {len(self.chunks)}")

        # Stats by portal
        table = Table(title="Chunks by Portal")
        table.add_column("Portal", style="cyan")
        table.add_column("Chunks", justify="right", style="green")

        portal_counts = {}
        for chunk in self.chunks:
            portal_counts[chunk.portal] = portal_counts.get(chunk.portal, 0) + 1

        for portal, count in sorted(portal_counts.items(), key=lambda x: -x[1]):
            table.add_row(portal, str(count))

        console.print(table)

        # Stats by section
        table2 = Table(title="Chunks by Section")
        table2.add_column("Section", style="cyan")
        table2.add_column("Chunks", justify="right", style="green")

        section_counts = {}
        for chunk in self.chunks:
            section_counts[chunk.section] = section_counts.get(chunk.section, 0) + 1

        for section, count in sorted(section_counts.items(), key=lambda x: -x[1]):
            table2.add_row(section, str(count))

        console.print(table2)

        # Hindi content stats
        hindi_chunks = sum(1 for c in self.chunks if c.metadata.get('has_hindi', False))
        console.print(f"\n[cyan]Hindi-content chunks: {hindi_chunks}[/cyan]")

        return self.chunks

    def save_chunks(self):
        """Save chunks to JSON"""
        output_file = CHUNKS_DIR / "all_chunks.json"
        data = [asdict(chunk) for chunk in self.chunks]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        console.print(f"\n[blue]Chunks saved to {output_file}[/blue]")


# Backwards compatibility alias
DocumentChunker = SemanticChunker


def main():
    chunker = SemanticChunker()
    chunker.chunk_all()


if __name__ == "__main__":
    main()
