"""
Fast PDF Parser using PyMuPDF - No API limits, Python 3.14 compatible
Processes all PDFs locally with high speed
"""
import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
# Sequential processing - fitz doesn't work well with threads

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DIR, PROCESSED_DIR, LOGS_DIR

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


def extract_text_from_pdf(file_path: Path) -> Optional[tuple]:
    """Extract text from a single PDF using PyMuPDF"""
    try:
        doc = fitz.open(str(file_path))
        text_parts = []
        num_pages = len(doc)

        for page_num in range(num_pages):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                text_parts.append(text)

        doc.close()

        full_text = "\n\n".join(text_parts)

        # Skip if very little content
        if len(full_text) < 50:
            return None

        return (full_text, num_pages)

    except Exception as e:
        return None


def parse_all_pdfs() -> Dict[str, ParsedDocument]:
    """Parse all PDFs using PyMuPDF"""
    console.print("\n[bold blue]========================================[/bold blue]")
    console.print("[bold blue]  Fast PDF Parser (PyMuPDF)[/bold blue]")
    console.print("[bold blue]========================================[/bold blue]\n")

    # Find all PDF files
    pdf_files = []

    # Portal PDFs
    if (RAW_DIR / "pdf").exists():
        portal_pdfs = list((RAW_DIR / "pdf").glob("*.pdf"))
        pdf_files.extend(portal_pdfs)
        console.print(f"[cyan]Found {len(portal_pdfs)} portal PDFs[/cyan]")

    # Shasanadesh GOs
    if (RAW_DIR / "shasanadesh").exists():
        shasanadesh_pdfs = list((RAW_DIR / "shasanadesh").glob("*.pdf"))
        pdf_files.extend(shasanadesh_pdfs)
        console.print(f"[cyan]Found {len(shasanadesh_pdfs)} Shasanadesh GOs[/cyan]")

    if not pdf_files:
        console.print("[yellow]No PDF files found[/yellow]")
        return {}

    console.print(f"\n[blue]Total: {len(pdf_files)} PDFs to parse[/blue]")

    # Check for existing parsed docs to resume
    parsed_file = PROCESSED_DIR / "parsed_documents.json"
    parsed_docs = {}

    if parsed_file.exists():
        try:
            with open(parsed_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            parsed_docs = {k: ParsedDocument(**v) for k, v in existing.items()}
            console.print(f"[green]Loaded {len(parsed_docs)} existing parsed documents[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not load existing: {e}[/yellow]")

    # Filter out already parsed
    remaining_files = [f for f in pdf_files if str(f) not in parsed_docs]
    console.print(f"[blue]Parsing {len(remaining_files)} remaining documents[/blue]\n")

    if not remaining_files:
        console.print("[green]All documents already parsed![/green]")
        return parsed_docs

    # Parse with progress bar (sequential - fitz doesn't play well with threads)
    failed_files = []
    success_count = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Parsing PDFs...", total=len(remaining_files))

        for file_path in remaining_files:
            try:
                result = extract_text_from_pdf(file_path)

                if result:
                    full_text, num_pages = result

                    # Determine portal and URL
                    if "shasanadesh" in str(file_path):
                        portal = "shasanadesh.up.gov.in"
                        url = f"https://shasanadesh.up.gov.in/GO/{file_path.name}"
                    else:
                        portal = "invest.up.gov.in"
                        url = ""

                    parsed_doc = ParsedDocument(
                        file_path=str(file_path),
                        original_url=url,
                        portal=portal,
                        text=full_text,
                        num_pages=num_pages,
                        doc_type="pdf",
                        language="hi",  # Most are Hindi
                        parsed_at=datetime.now().isoformat(),
                        metadata={
                            "filename": file_path.name,
                            "file_size": file_path.stat().st_size,
                            "char_count": len(full_text),
                        }
                    )

                    parsed_docs[str(file_path)] = parsed_doc
                    success_count += 1
                else:
                    failed_files.append(str(file_path))

            except Exception as e:
                failed_files.append(str(file_path))

            progress.update(task, advance=1)

            # Save progress every 100 documents
            if success_count % 100 == 0 and success_count > 0:
                save_results(parsed_docs)

    # Final save
    save_results(parsed_docs)

    # Summary
    console.print(f"\n[bold green]Parsing Summary:[/bold green]")
    console.print(f"  Successfully parsed: {success_count}")
    console.print(f"  Failed: {len(failed_files)}")
    console.print(f"  Total parsed: {len(parsed_docs)}")

    total_chars = sum(len(doc.text) for doc in parsed_docs.values())
    console.print(f"  Total characters: {total_chars:,}")

    # By portal
    by_portal = {}
    for doc in parsed_docs.values():
        by_portal[doc.portal] = by_portal.get(doc.portal, 0) + 1
    console.print("\n[bold]Parsed by portal:[/bold]")
    for portal, count in sorted(by_portal.items()):
        console.print(f"  {portal}: {count}")

    if failed_files:
        failed_file = LOGS_DIR / "failed_parsing.json"
        with open(failed_file, "w") as f:
            json.dump(failed_files, f, indent=2)
        console.print(f"\n[yellow]Failed files saved to {failed_file}[/yellow]")

    return parsed_docs


def save_results(parsed_docs: Dict[str, ParsedDocument]):
    """Save parsed documents"""
    output_file = PROCESSED_DIR / "parsed_documents.json"
    data = {path: asdict(doc) for path, doc in parsed_docs.items()}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parse_all_pdfs()


if __name__ == "__main__":
    main()
