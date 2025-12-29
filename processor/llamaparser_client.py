"""
LlamaParser integration for multi-portal PDF document processing
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import time

from llama_parse import LlamaParse
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import LLAMA_PARSER_API_KEY, RAW_DIR, PROCESSED_DIR, LOGS_DIR

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


class LlamaParserClient:
    """Process documents using LlamaParser API"""

    def __init__(self):
        self.parser = LlamaParse(
            api_key=LLAMA_PARSER_API_KEY,
            result_type="markdown",
            parsing_instruction="""
            Extract all text content from this document accurately.
            This is an official government document from Uttar Pradesh, India related to investment and industry.

            It may contain:
            - Policy details, incentives, and schemes
            - Government orders and circulars
            - Tables with financial data, rates, and statistics
            - Hindi text (preserve as-is or transliterate)
            - Legal/official language
            - Procedures and checklists
            - Contact information and deadlines

            Important:
            - Preserve all numerical data exactly (percentages, amounts, dates)
            - Keep table structures intact with proper formatting
            - Extract all headings and subheadings
            - Include reference numbers, GO numbers, and dates
            - Preserve any contact information (phone, email, addresses)
            """,
            language="en",
            verbose=False
        )
        self.parsed_docs: Dict[str, ParsedDocument] = {}
        self.failed_files: List[str] = []

    async def parse_document(self, file_path: Path, doc_info: dict) -> Optional[ParsedDocument]:
        """Parse a single document"""
        try:
            # LlamaParse works synchronously
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(
                None,
                lambda: self.parser.load_data(str(file_path))
            )

            if not documents:
                console.print(f"[yellow]No content extracted from {file_path.name}[/yellow]")
                return None

            # Combine all pages
            full_text = "\n\n".join([doc.text for doc in documents])

            if len(full_text) < 50:
                console.print(f"[yellow]Very little content from {file_path.name}[/yellow]")
                return None

            return ParsedDocument(
                file_path=str(file_path),
                original_url=doc_info.get("url", ""),
                portal=doc_info.get("portal", "unknown"),
                text=full_text,
                num_pages=len(documents),
                doc_type=file_path.suffix.lstrip("."),
                language="en",
                parsed_at=datetime.now().isoformat(),
                metadata={
                    "filename": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "char_count": len(full_text),
                    "category": doc_info.get("category", ""),
                    "title": doc_info.get("title", ""),
                }
            )

        except Exception as e:
            console.print(f"[red]Error parsing {file_path.name}: {e}[/red]")
            self.failed_files.append(str(file_path))
            return None

    async def parse_all_documents(self) -> Dict[str, ParsedDocument]:
        """Parse all downloaded documents"""
        console.print("\n[bold blue]========================================[/bold blue]")
        console.print("[bold blue]  LlamaParser Document Processing[/bold blue]")
        console.print("[bold blue]========================================[/bold blue]\n")

        # Load download results for metadata
        path_to_info = {}
        download_file = LOGS_DIR / "downloaded_documents.json"
        if download_file.exists():
            with open(download_file, "r") as f:
                all_docs = json.load(f)
            for url, doc_info in all_docs.items():
                local_path = doc_info.get("local_path")
                if local_path and Path(local_path).exists():
                    path_to_info[local_path] = doc_info

        # Find all PDF files across all directories
        pdf_files = []

        # Portal PDFs in data/raw/pdf/
        if (RAW_DIR / "pdf").exists():
            pdf_files.extend((RAW_DIR / "pdf").glob("*.pdf"))
            console.print(f"[cyan]Found {len(list((RAW_DIR / 'pdf').glob('*.pdf')))} portal PDFs[/cyan]")

        # Shasanadesh GOs in data/raw/shasanadesh/
        if (RAW_DIR / "shasanadesh").exists():
            shasanadesh_pdfs = list((RAW_DIR / "shasanadesh").glob("*.pdf"))
            pdf_files.extend(shasanadesh_pdfs)
            console.print(f"[cyan]Found {len(shasanadesh_pdfs)} Shasanadesh GOs[/cyan]")

            # Add default metadata for Shasanadesh files
            for pdf in shasanadesh_pdfs:
                if str(pdf) not in path_to_info:
                    path_to_info[str(pdf)] = {
                        "url": f"https://shasanadesh.up.gov.in/GO/{pdf.name}",
                        "portal": "shasanadesh.up.gov.in",
                        "category": "government_order",
                        "title": pdf.stem
                    }

        # Portal subdirectories (old structure)
        for portal_dir in RAW_DIR.iterdir():
            if portal_dir.is_dir() and portal_dir.name not in ["pdf", "shasanadesh"]:
                for type_dir in portal_dir.iterdir():
                    if type_dir.is_dir():
                        pdf_files.extend(type_dir.glob("*.pdf"))

        if not pdf_files:
            console.print("[yellow]No PDF files found to parse[/yellow]")
            return {}

        console.print(f"[blue]Found {len(pdf_files)} PDF files to parse[/blue]")

        # Check for existing parsed docs to resume
        parsed_file = PROCESSED_DIR / "parsed_documents.json"
        if parsed_file.exists():
            with open(parsed_file, "r") as f:
                existing = json.load(f)
            self.parsed_docs = {k: ParsedDocument(**v) for k, v in existing.items()}
            console.print(f"[green]Loaded {len(self.parsed_docs)} existing parsed documents[/green]")

        # Filter out already parsed
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
                doc_info = path_to_info.get(str(file_path), {})

                parsed = await self.parse_document(file_path, doc_info)

                if parsed:
                    self.parsed_docs[str(file_path)] = parsed
                    console.print(f"[green]Parsed: {file_path.name} ({parsed.num_pages} pages, {len(parsed.text)} chars)[/green]")

                progress.update(task, advance=1)

                # Save progress every 10 documents
                if len(self.parsed_docs) % 10 == 0:
                    await self.save_results()

                # Rate limiting for LlamaParser API (free tier has limits)
                await asyncio.sleep(2)

        # Final save
        await self.save_results()

        # Summary
        console.print(f"\n[bold green]Parsing Summary:[/bold green]")
        console.print(f"  Successfully parsed: {len(self.parsed_docs)}")
        console.print(f"  Failed: {len(self.failed_files)}")

        total_chars = sum(len(doc.text) for doc in self.parsed_docs.values())
        console.print(f"  Total characters: {total_chars:,}")

        # By portal
        by_portal = {}
        for doc in self.parsed_docs.values():
            by_portal[doc.portal] = by_portal.get(doc.portal, 0) + 1
        console.print("\n[bold]Parsed by portal:[/bold]")
        for portal, count in sorted(by_portal.items()):
            console.print(f"  {portal}: {count}")

        if self.failed_files:
            failed_file = LOGS_DIR / "failed_parsing.json"
            with open(failed_file, "w") as f:
                json.dump(self.failed_files, f, indent=2)
            console.print(f"\n[yellow]Failed files saved to {failed_file}[/yellow]")

        return self.parsed_docs

    async def save_results(self):
        """Save parsed documents"""
        output_file = PROCESSED_DIR / "parsed_documents.json"
        data = {path: asdict(doc) for path, doc in self.parsed_docs.items()}

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Also save individual markdown files
        md_dir = PROCESSED_DIR / "markdown"
        md_dir.mkdir(exist_ok=True)

        for path, doc in self.parsed_docs.items():
            safe_name = Path(path).stem[:50]
            md_file = md_dir / f"{safe_name}.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(f"# {Path(path).name}\n\n")
                f.write(f"**Portal:** {doc.portal}\n")
                f.write(f"**Source:** {doc.original_url}\n")
                f.write(f"**Pages:** {doc.num_pages}\n\n")
                f.write("---\n\n")
                f.write(doc.text)


async def main():
    client = LlamaParserClient()
    await client.parse_all_documents()


if __name__ == "__main__":
    asyncio.run(main())
