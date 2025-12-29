"""
Extract text content from HTML pages for RAG
"""
import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from urllib.parse import urlparse

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import BASE_URL, PROCESSED_DIR, LOGS_DIR

console = Console()


@dataclass
class HTMLContent:
    """Extracted content from an HTML page"""
    url: str
    title: str
    content: str
    headings: List[str]
    section: str  # e.g., 'policy', 'sector', 'faq'
    extracted_at: str
    metadata: Dict


class HTMLExtractor:
    """Extract clean text content from HTML pages"""

    # Pages to extract content from
    CONTENT_PAGES = [
        # Why UP section
        ("/economic-snapshot/", "why_up"),
        ("/demography/", "why_up"),
        ("/infrastructure/", "why_up"),
        ("/business-climate/", "why_up"),

        # FAQs
        ("/faqs/", "faq"),

        # Policies (detail pages will be discovered)
        ("/policies/", "policy"),

        # About pages
        ("/about-invest-up/", "about"),
        ("/gold-card-scheme/", "scheme"),
        ("/nivesh-mitra-single-window-clearance/", "service"),
        ("/udyami-mitra/", "service"),

        # Sectors
        ("/retail-e-commerce/", "sector"),
        ("/defence-aerospace-sector/", "sector"),
        ("/it-ites/", "sector"),
        ("/startup-sector/", "sector"),
        ("/tourism-hospitality/", "sector"),
        ("/msme-sector/", "sector"),
        ("/electronics-technology/", "sector"),
        ("/handloom-and-textile-sector/", "sector"),
        ("/renewable-energy-sector/", "sector"),
        ("/warehousing-logistics-sector/", "sector"),
        ("/auto-components-sector/", "sector"),
        ("/medical-device-and-pharma/", "sector"),
        ("/agriculture-food-processing/", "sector"),
        ("/fintech-sector/", "sector"),
        ("/dairy-sector/", "sector"),
        ("/ev-manufacturing-sector/", "sector"),
        ("/semiconductor-sector/", "sector"),
        ("/film-sector/", "sector"),
        ("/urban-development/", "sector"),
        ("/civil-aviation-sector/", "sector"),
    ]

    def __init__(self):
        self.extracted_content: Dict[str, HTMLContent] = {}

    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove script/style content remnants
        text = re.sub(r'<[^>]+>', '', text)
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text.strip()

    def extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content, removing navigation, footer, etc."""
        # Remove unwanted elements
        for element in soup.find_all(['nav', 'footer', 'header', 'script', 'style', 'aside']):
            element.decompose()

        # Remove elements with common navigation/footer classes
        for selector in ['.menu', '.navigation', '.footer', '.sidebar', '.breadcrumb', '.social-share']:
            for element in soup.select(selector):
                element.decompose()

        # Try to find main content area
        main_content = (
            soup.find('main') or
            soup.find('article') or
            soup.find(class_=re.compile(r'content|main|entry|post')) or
            soup.find(id=re.compile(r'content|main|entry|post')) or
            soup.body
        )

        if main_content:
            return self.clean_text(main_content.get_text(separator=' '))
        return ""

    def extract_headings(self, soup: BeautifulSoup) -> List[str]:
        """Extract all headings for structure"""
        headings = []
        for h in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            text = self.clean_text(h.get_text())
            if text and len(text) > 2:
                headings.append(text)
        return headings

    async def extract_page(self, page, url: str, section: str) -> Optional[HTMLContent]:
        """Extract content from a single page"""
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(0.5)

            html = await page.content()
            soup = BeautifulSoup(html, "lxml")

            title = await page.title()
            content = self.extract_main_content(soup)
            headings = self.extract_headings(soup)

            if len(content) < 100:  # Skip mostly empty pages
                return None

            return HTMLContent(
                url=url,
                title=title,
                content=content,
                headings=headings,
                section=section,
                extracted_at=datetime.now().isoformat(),
                metadata={
                    "content_length": len(content),
                    "heading_count": len(headings)
                }
            )

        except Exception as e:
            console.print(f"[red]Error extracting {url}: {e}[/red]")
            return None

    async def extract_all(self) -> Dict[str, HTMLContent]:
        """Extract content from all relevant pages"""
        console.print("[bold blue]Extracting HTML content...[/bold blue]")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = await context.new_page()

            with Progress() as progress:
                task = progress.add_task("Extracting...", total=len(self.CONTENT_PAGES))

                for path, section in self.CONTENT_PAGES:
                    url = f"{BASE_URL}{path}"
                    content = await self.extract_page(page, url, section)

                    if content:
                        self.extracted_content[url] = content
                        console.print(f"[green]Extracted: {url} ({len(content.content)} chars)[/green]")

                    progress.update(task, advance=1)

            await browser.close()

        # Save results
        await self.save_results()

        console.print(f"\n[bold green]Extraction Summary:[/bold green]")
        console.print(f"  Pages extracted: {len(self.extracted_content)}")

        return self.extracted_content

    async def save_results(self):
        """Save extracted content to JSON"""
        output_file = PROCESSED_DIR / "html_content.json"

        data = {url: asdict(content) for url, content in self.extracted_content.items()}

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        console.print(f"[blue]HTML content saved to {output_file}[/blue]")


async def main():
    extractor = HTMLExtractor()
    await extractor.extract_all()


if __name__ == "__main__":
    asyncio.run(main())
