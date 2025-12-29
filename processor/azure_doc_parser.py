"""
Azure Document Intelligence Parser with Mistral Fallback
Tier 1: Azure Document Intelligence (high-accuracy OCR)
Tier 2: Mistral Document AI (fallback for failed documents)
"""
import json
import time
import base64
import requests
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional
import fitz  # PyMuPDF for PDF to image conversion

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    AZURE_DOC_INTELLIGENCE_ENDPOINT,
    AZURE_DOC_INTELLIGENCE_KEY,
    MISTRAL_DOC_AI_ENDPOINT,
    MISTRAL_DOC_AI_KEY,
    RAW_DIR,
    PROCESSED_DIR,
    LOGS_DIR
)

console = Console()


@dataclass
class ParsedDocument:
    """Parsed document with extracted content"""
    file_path: str
    original_url: str
    portal: str
    text: str
    num_pages: int
    doc_type: str
    language: str
    parsed_at: str
    metadata: Dict


class AzureDocumentParser:
    """Process documents using Azure Document Intelligence with Mistral fallback"""

    def __init__(self):
        # Initialize Azure Document Intelligence client
        if AZURE_DOC_INTELLIGENCE_ENDPOINT and AZURE_DOC_INTELLIGENCE_KEY:
            self.azure_client = DocumentAnalysisClient(
                endpoint=AZURE_DOC_INTELLIGENCE_ENDPOINT,
                credential=AzureKeyCredential(AZURE_DOC_INTELLIGENCE_KEY)
            )
            console.print("[green]Azure Document Intelligence initialized[/green]")
        else:
            self.azure_client = None
            console.print("[yellow]Azure Document Intelligence not configured[/yellow]")

        # Mistral fallback config
        self.mistral_endpoint = MISTRAL_DOC_AI_ENDPOINT
        self.mistral_key = MISTRAL_DOC_AI_KEY
        if self.mistral_endpoint and self.mistral_key:
            console.print("[green]Mistral Document AI fallback configured[/green]")
        else:
            console.print("[yellow]Mistral fallback not configured[/yellow]")

        self.parsed_docs: Dict[str, ParsedDocument] = {}
        self.failed_files: List[str] = []
        self.fallback_used: List[str] = []

    def parse_with_azure(self, file_path: Path) -> Optional[str]:
        """Parse document using Azure Document Intelligence"""
        try:
            with open(file_path, "rb") as f:
                poller = self.azure_client.begin_analyze_document(
                    "prebuilt-read",
                    document=f
                )
                result = poller.result()

            if not result.content or len(result.content.strip()) < 50:
                return None

            # Extract text with structure
            return self._extract_structured_text(result)

        except Exception as e:
            console.print(f"[yellow]Azure failed for {file_path.name}: {e}[/yellow]")
            return None

    def parse_with_mistral(self, file_path: Path) -> Optional[str]:
        """Parse document using Mistral Document AI as fallback"""
        if not self.mistral_endpoint or not self.mistral_key:
            return None

        try:
            # Convert PDF pages to images and send to Mistral
            doc = fitz.open(file_path)
            all_text = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                # Render page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img_bytes = pix.tobytes("png")
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")

                # Call Mistral OCR API
                headers = {
                    "Content-Type": "application/json",
                    "api-key": self.mistral_key
                }

                payload = {
                    "model": "mistral-document-ai-2505",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract all text from this document image. Preserve the structure including tables, headings, and lists. Return only the extracted text, no explanations."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 4096
                }

                response = requests.post(
                    self.mistral_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    page_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    all_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
                else:
                    console.print(f"[yellow]Mistral API error on page {page_num + 1}: {response.status_code}[/yellow]")

            doc.close()

            if all_text:
                return "\n\n".join(all_text)
            return None

        except Exception as e:
            console.print(f"[red]Mistral fallback failed for {file_path.name}: {e}[/red]")
            return None

    def parse_document(self, file_path: Path, doc_info: dict) -> Optional[ParsedDocument]:
        """Parse a single document using tiered approach"""
        # Try Azure first
        text = None
        parser_used = None

        if self.azure_client:
            text = self.parse_with_azure(file_path)
            if text:
                parser_used = "azure_document_intelligence"

        # Fallback to Mistral if Azure fails
        if not text and self.mistral_endpoint:
            console.print(f"[yellow]Using Mistral fallback for {file_path.name}[/yellow]")
            text = self.parse_with_mistral(file_path)
            if text:
                parser_used = "mistral_document_ai"
                self.fallback_used.append(str(file_path))

        if not text or len(text) < 50:
            console.print(f"[red]All parsers failed for {file_path.name}[/red]")
            self.failed_files.append(str(file_path))
            return None

        # Count pages
        try:
            doc = fitz.open(file_path)
            num_pages = len(doc)
            doc.close()
        except:
            num_pages = 1

        return ParsedDocument(
            file_path=str(file_path),
            original_url=doc_info.get("url", ""),
            portal=doc_info.get("portal", "unknown"),
            text=text,
            num_pages=num_pages,
            doc_type=file_path.suffix.lstrip("."),
            language="mixed",
            parsed_at=datetime.now().isoformat(),
            metadata={
                "filename": file_path.name,
                "file_size": file_path.stat().st_size,
                "char_count": len(text),
                "category": doc_info.get("category", ""),
                "title": doc_info.get("title", ""),
                "parser": parser_used
            }
        )

    def _extract_structured_text(self, result) -> str:
        """Extract text preserving structure from Azure result"""
        lines = []

        for page in result.pages:
            page_lines = []
            for line in page.lines:
                page_lines.append(line.content)
            lines.extend(page_lines)
            lines.append("\n---\n")

        # Process tables
        if result.tables:
            lines.append("\n## Tables\n")
            for i, table in enumerate(result.tables):
                lines.append(f"\n### Table {i+1}\n")
                table_data = {}
                for cell in table.cells:
                    row_idx = cell.row_index
                    col_idx = cell.column_index
                    if row_idx not in table_data:
                        table_data[row_idx] = {}
                    table_data[row_idx][col_idx] = cell.content

                if table_data:
                    max_cols = max(max(cols.keys()) for cols in table_data.values()) + 1
                    for row_idx in sorted(table_data.keys()):
                        row = table_data[row_idx]
                        row_text = " | ".join(row.get(c, "") for c in range(max_cols))
                        lines.append(f"| {row_text} |")
                        if row_idx == 0:
                            lines.append("|" + "---|" * max_cols)

        return "\n".join(lines)

    def save_results(self):
        """Save parsed documents"""
        output_file = PROCESSED_DIR / "parsed_documents.json"
        data = {path: asdict(doc) for path, doc in self.parsed_docs.items()}

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Save markdown files
        md_dir = PROCESSED_DIR / "markdown"
        md_dir.mkdir(exist_ok=True)

        for path, doc in self.parsed_docs.items():
            safe_name = Path(path).stem[:50]
            md_file = md_dir / f"{safe_name}.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(f"# {Path(path).name}\n\n")
                f.write(f"**Portal:** {doc.portal}\n")
                f.write(f"**Source:** {doc.original_url}\n")
                f.write(f"**Pages:** {doc.num_pages}\n")
                f.write(f"**Parser:** {doc.metadata.get('parser', 'unknown')}\n\n")
                f.write("---\n\n")
                f.write(doc.text)

    def parse_all_documents(self) -> Dict[str, ParsedDocument]:
        """Parse all remaining documents"""
        console.print("\n[bold blue]========================================[/bold blue]")
        console.print("[bold blue]  Document Parser (Azure + Mistral)[/bold blue]")
        console.print("[bold blue]========================================[/bold blue]\n")

        # Load metadata
        path_to_info = {}
        download_file = LOGS_DIR / "downloaded_documents.json"
        if download_file.exists():
            with open(download_file, "r") as f:
                all_docs = json.load(f)
            for url, doc_info in all_docs.items():
                local_path = doc_info.get("local_path")
                if local_path and Path(local_path).exists():
                    path_to_info[local_path] = doc_info

        # Find all PDF files
        pdf_files = []

        # Scan all subdirectories for PDFs
        all_pdfs = list(RAW_DIR.rglob("*.pdf"))
        console.print(f"[cyan]Found {len(all_pdfs)} total PDFs in all directories[/cyan]")

        # Deduplicate by content hash (keep first occurrence)
        import hashlib
        seen_hashes = set()
        for pdf in all_pdfs:
            try:
                with open(pdf, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash not in seen_hashes:
                    seen_hashes.add(file_hash)
                    pdf_files.append(pdf)

                    # Add metadata for files not in path_to_info
                    if str(pdf) not in path_to_info:
                        portal = "invest.up.gov.in"
                        category = "policy"
                        if "shasanadesh" in str(pdf).lower():
                            portal = "shasanadesh.up.gov.in"
                            category = "government_order"
                        elif "niveshmitra" in str(pdf).lower():
                            portal = "niveshmitra.up.nic.in"
                        elif "upeida" in str(pdf).lower():
                            portal = "upeida.up.gov.in"
                        elif "startinup" in str(pdf).lower():
                            portal = "startinup.up.gov.in"
                        elif "niveshsarathi" in str(pdf).lower():
                            portal = "niveshsarathi.up.gov.in"

                        path_to_info[str(pdf)] = {
                            "url": f"https://{portal}/pdf/{pdf.name}",
                            "portal": portal,
                            "category": category,
                            "title": pdf.stem
                        }
            except Exception as e:
                console.print(f"[yellow]Error reading {pdf.name}: {e}[/yellow]")

        console.print(f"[cyan]After deduplication: {len(pdf_files)} unique PDFs[/cyan]")

        if not pdf_files:
            console.print("[yellow]No PDF files found[/yellow]")
            return {}

        console.print(f"[blue]Total: {len(pdf_files)} PDF files[/blue]")

        # Load existing parsed docs
        parsed_file = PROCESSED_DIR / "parsed_documents.json"
        if parsed_file.exists():
            try:
                with open(parsed_file, "r") as f:
                    existing = json.load(f)
                self.parsed_docs = {k: ParsedDocument(**v) for k, v in existing.items()}
                console.print(f"[green]Loaded {len(self.parsed_docs)} existing parsed documents[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not load existing docs: {e}[/yellow]")

        # Filter already parsed
        remaining_files = [f for f in pdf_files if str(f) not in self.parsed_docs]
        console.print(f"[blue]Parsing {len(remaining_files)} remaining documents[/blue]")

        if not remaining_files:
            console.print("[green]All documents already parsed![/green]")
            return self.parsed_docs

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Parsing PDFs...", total=len(remaining_files))

            for file_path in remaining_files:
                doc_info = path_to_info.get(str(file_path), {
                    "url": "",
                    "portal": "invest.up.gov.in" if "pdf" in str(file_path) else "shasanadesh.up.gov.in",
                    "category": "policy",
                    "title": file_path.stem
                })

                parsed = self.parse_document(file_path, doc_info)

                if parsed:
                    self.parsed_docs[str(file_path)] = parsed
                    parser = parsed.metadata.get("parser", "unknown")
                    console.print(f"[green]Parsed [{parser}]: {file_path.name} ({parsed.num_pages} pages)[/green]")

                progress.update(task, advance=1)

                # Save every 10 docs
                if len(self.parsed_docs) % 10 == 0:
                    self.save_results()

                time.sleep(0.2)

        # Final save
        self.save_results()

        # Summary
        console.print(f"\n[bold green]Parsing Summary:[/bold green]")
        console.print(f"  Successfully parsed: {len(self.parsed_docs)}")
        console.print(f"  Failed: {len(self.failed_files)}")
        console.print(f"  Used Mistral fallback: {len(self.fallback_used)}")

        total_chars = sum(len(doc.text) for doc in self.parsed_docs.values())
        total_pages = sum(doc.num_pages for doc in self.parsed_docs.values())
        console.print(f"  Total pages: {total_pages:,}")
        console.print(f"  Total characters: {total_chars:,}")

        # By parser
        by_parser = {}
        for doc in self.parsed_docs.values():
            parser = doc.metadata.get("parser", "unknown")
            by_parser[parser] = by_parser.get(parser, 0) + 1
        console.print("\n[bold]Parsed by method:[/bold]")
        for parser, count in sorted(by_parser.items()):
            console.print(f"  {parser}: {count}")

        if self.failed_files:
            failed_file = LOGS_DIR / "failed_parsing.json"
            with open(failed_file, "w") as f:
                json.dump(self.failed_files, f, indent=2)
            console.print(f"\n[yellow]Failed files saved to {failed_file}[/yellow]")

        return self.parsed_docs


def main():
    parser = AzureDocumentParser()
    parser.parse_all_documents()


if __name__ == "__main__":
    main()
