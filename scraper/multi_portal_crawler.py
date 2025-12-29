"""
Comprehensive Multi-Portal Crawler for UP Investment Ecosystem
Crawls all 6 portals for 100% coverage
"""
import asyncio
import json
import re
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
from dataclasses import dataclass, field, asdict
from typing import Set, Dict, List, Optional, Tuple
from datetime import datetime
from abc import ABC, abstractmethod

from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
import aiohttp
import aiofiles

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DIR, LOGS_DIR, DOCUMENT_EXTENSIONS, REQUEST_DELAY

console = Console()


@dataclass
class DocumentInfo:
    """Information about a discovered document"""
    url: str
    source_page: str
    portal: str
    doc_type: str
    title: Optional[str] = None
    category: Optional[str] = None
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    downloaded: bool = False
    local_path: Optional[str] = None
    file_hash: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class PortalCrawler(ABC):
    """Base class for portal-specific crawlers"""

    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url
        self.visited_urls: Set[str] = set()
        self.document_urls: Dict[str, DocumentInfo] = {}
        self.page_content: Dict[str, str] = {}  # URL -> extracted text

    @abstractmethod
    async def crawl(self, browser: Browser, progress: Progress) -> Dict[str, DocumentInfo]:
        """Crawl the portal and return discovered documents"""
        pass

    def normalize_url(self, url: str) -> Optional[str]:
        """Normalize URL"""
        if not url or url.startswith(('javascript:', 'mailto:', 'tel:', '#')):
            return None

        # Handle relative URLs
        if url.startswith('/'):
            url = urljoin(self.base_url, url)
        elif not url.startswith('http'):
            url = urljoin(self.base_url, url)

        # Normalize http to https where appropriate
        if 'up.gov.in' in url or 'up.nic.in' in url:
            url = url.replace('http://', 'https://')

        # Remove fragments
        url = url.split('#')[0]

        return url

    def is_document_url(self, url: str) -> bool:
        """Check if URL is a document"""
        if not url:
            return False
        parsed = urlparse(url.lower())
        return any(parsed.path.endswith(ext) for ext in DOCUMENT_EXTENSIONS)

    def get_doc_type(self, url: str) -> str:
        """Get document type from URL"""
        parsed = urlparse(url.lower())
        for ext in DOCUMENT_EXTENSIONS:
            if parsed.path.endswith(ext):
                return ext.lstrip('.')
        return 'unknown'

    async def extract_links(self, page: Page) -> Set[str]:
        """Extract all links from page"""
        links = set()
        try:
            content = await page.content()
            soup = BeautifulSoup(content, 'lxml')

            # Standard href links
            for a in soup.find_all('a', href=True):
                url = self.normalize_url(a['href'])
                if url:
                    links.add(url)

            # iframe sources
            for iframe in soup.find_all('iframe', src=True):
                url = self.normalize_url(iframe['src'])
                if url:
                    links.add(url)

            # Data attributes
            for attr in ['data-href', 'data-url', 'data-pdf', 'data-src']:
                for tag in soup.find_all(attrs={attr: True}):
                    url = self.normalize_url(tag[attr])
                    if url:
                        links.add(url)

            # JavaScript patterns
            scripts = soup.find_all('script')
            js_patterns = [
                r'["\']([^"\']+\.pdf)["\']',
                r'["\']([^"\']+\.xlsx?)["\']',
                r'["\']([^"\']+\.docx?)["\']',
                r'href\s*[=:]\s*["\']([^"\']+)["\']',
            ]
            for script in scripts:
                if script.string:
                    for pattern in js_patterns:
                        matches = re.findall(pattern, script.string)
                        for match in matches:
                            url = self.normalize_url(match)
                            if url:
                                links.add(url)

        except Exception as e:
            console.print(f"[yellow]Error extracting links: {e}[/yellow]")

        return links

    def add_document(self, url: str, source_page: str, title: str = None, category: str = None):
        """Add a discovered document"""
        if url not in self.document_urls:
            self.document_urls[url] = DocumentInfo(
                url=url,
                source_page=source_page,
                portal=self.name,
                doc_type=self.get_doc_type(url),
                title=title,
                category=category
            )
            console.print(f"[green][{self.name}] Found: {url}[/green]")


class InvestUPCrawler(PortalCrawler):
    """Crawler for invest.up.gov.in"""

    def __init__(self):
        super().__init__("invest.up.gov.in", "https://invest.up.gov.in")

        self.priority_pages = [
            "/", "/policies/", "/gos/", "/circulars/", "/circulars-archive/",
            "/faqs/", "/economic-snapshot/", "/demography/", "/infrastructure/",
            "/business-climate/", "/newsletter/", "/tender/", "/rti/",
            "/model-templates/", "/notifications/", "/press-releases/",
            "/investible-projects/", "/manpower/", "/know-your-approvals/",
            "/about-invest-up/", "/gold-card-scheme/", "/helpdesk/",
            "/nivesh-mitra-single-window-clearance/", "/udyami-mitra/",
        ]

        self.sector_pages = [
            "/retail-e-commerce/", "/defence-aerospace-sector/", "/it-ites/",
            "/startup-sector/", "/tourism-hospitality/", "/msme-sector/",
            "/electronics-technology/", "/handloom-and-textile-sector/",
            "/renewable-energy-sector/", "/warehousing-logistics-sector/",
            "/auto-components-sector/", "/medical-device-and-pharma/",
            "/agriculture-food-processing/", "/fintech-sector/", "/dairy-sector/",
            "/ev-manufacturing-sector/", "/semiconductor-sector/", "/film-sector/",
            "/urban-development/", "/civil-aviation-sector/",
        ]

        self.event_pages = [
            "/ground-breaking-ceremony-4-0/", "/upgis-2023-roadshows/",
            "/up-international-trade-show-2024/", "/motogp-2023/",
            "/ev-roundtable-conference/", "/up-investors-summit-2018/",
        ]

    async def crawl(self, browser: Browser, progress: Progress) -> Dict[str, DocumentInfo]:
        """Crawl invest.up.gov.in"""
        task = progress.add_task(f"[cyan]{self.name}", total=100)

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = await context.new_page()

        # Collect all pages to crawl
        all_pages = self.priority_pages + self.sector_pages + self.event_pages

        # Add paginated GO pages
        for i in range(1, 40):  # Up to 40 pages to be safe
            all_pages.append(f"/gos/?pagenum={i}")

        progress.update(task, total=len(all_pages))

        for page_path in all_pages:
            url = f"{self.base_url}{page_path}"
            if url in self.visited_urls:
                continue

            self.visited_urls.add(url)

            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(REQUEST_DELAY)

                # Extract links
                links = await self.extract_links(page)

                # Find documents
                for link in links:
                    if self.is_document_url(link):
                        title = await page.title()
                        self.add_document(link, url, title=title)
                    elif self.base_url in link and link not in self.visited_urls:
                        # Queue for crawling if it's a policy detail page
                        if any(p in link for p in ['/uttar-pradesh-', '/up-', '-policy']):
                            all_pages.append(link.replace(self.base_url, ''))

                # Extract page content for HTML pages
                content = await page.content()
                soup = BeautifulSoup(content, 'lxml')
                main_content = soup.find('main') or soup.find('article') or soup.body
                if main_content:
                    text = main_content.get_text(separator=' ', strip=True)
                    if len(text) > 100:
                        self.page_content[url] = text

            except Exception as e:
                console.print(f"[yellow]Error crawling {url}: {e}[/yellow]")

            progress.update(task, advance=1)

        await context.close()
        return self.document_urls


class NiveshMitraCrawler(PortalCrawler):
    """Crawler for niveshmitra.up.nic.in"""

    def __init__(self):
        super().__init__("niveshmitra.up.nic.in", "https://niveshmitra.up.nic.in")

        # Known direct document URLs
        self.direct_documents = [
            "Documents/Compendium.pdf",
            "Documents/PolicyHighlight.pdf",
            "Documents/IndustrialParks.pdf",
            "Documents/UM_InvestorRegProcess.pdf",
            "Documents/UM_CAF.pdf",
            "Documents/UM_Apply.pdf",
        ]

        # Pattern-based documents (179 services)
        self.pattern_documents = []
        for i in range(1, 180):
            self.pattern_documents.append(f"Documents/UPS_{i}.pdf")
            self.pattern_documents.append(f"Documents/DPF_{i}.pdf")

        # Government Orders (G_1 to G_70)
        for i in range(1, 71):
            self.pattern_documents.append(f"Documents/Govt_Orders/G_{i}.pdf")

        self.pages_to_crawl = [
            "/", "/About.aspx?ID=divcm", "/AdvantageUP.aspx",
            "/InvAssis.aspx?ID=chklst", "/Information.aspx?ID=1",
            "/UserManual.aspx", "/ServiceSOP.aspx", "/DashboardPublic.aspx",
            "/media.aspx?ID=1", "/media.aspx?ID=3",
        ]

    async def crawl(self, browser: Browser, progress: Progress) -> Dict[str, DocumentInfo]:
        """Crawl niveshmitra.up.nic.in"""
        total_items = len(self.direct_documents) + len(self.pattern_documents) + len(self.pages_to_crawl)
        task = progress.add_task(f"[cyan]{self.name}", total=total_items)

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = await context.new_page()

        # Add direct documents
        for doc_path in self.direct_documents:
            url = f"{self.base_url}/{doc_path}"
            self.add_document(url, self.base_url, category="core_document")
            progress.update(task, advance=1)

        # Add pattern-based documents
        for doc_path in self.pattern_documents:
            url = f"{self.base_url}/{doc_path}"
            self.add_document(url, self.base_url, category="process_flow")
            progress.update(task, advance=1)

        # Crawl pages for any additional documents
        for page_path in self.pages_to_crawl:
            url = f"{self.base_url}{page_path}"
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(REQUEST_DELAY)

                links = await self.extract_links(page)
                for link in links:
                    if self.is_document_url(link):
                        self.add_document(link, url)

                # Extract content
                content = await page.content()
                soup = BeautifulSoup(content, 'lxml')
                text = soup.get_text(separator=' ', strip=True)
                if len(text) > 100:
                    self.page_content[url] = text

            except Exception as e:
                console.print(f"[yellow]Error crawling {url}: {e}[/yellow]")

            progress.update(task, advance=1)

        await context.close()
        return self.document_urls


class NiveshSarathiCrawler(PortalCrawler):
    """Crawler for niveshsarathi.up.gov.in (with SSL bypass)"""

    def __init__(self):
        super().__init__("niveshsarathi.up.gov.in", "https://niveshsarathi.up.gov.in")

        self.pages_to_crawl = [
            "/investorcrm/",
            "/investorcrm/land_bank/upgis_land_bank",
            "/investorcrm/index.php/welcome/registration",
        ]

    async def crawl(self, browser: Browser, progress: Progress) -> Dict[str, DocumentInfo]:
        """Crawl niveshsarathi.up.gov.in with SSL bypass"""
        task = progress.add_task(f"[cyan]{self.name}", total=len(self.pages_to_crawl))

        # Create context with SSL bypass
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            ignore_https_errors=True  # Bypass SSL certificate errors
        )
        page = await context.new_page()

        for page_path in self.pages_to_crawl:
            url = f"{self.base_url}{page_path}"
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(REQUEST_DELAY)

                links = await self.extract_links(page)
                for link in links:
                    if self.is_document_url(link):
                        self.add_document(link, url)

                # Extract Land Bank data if available
                content = await page.content()
                soup = BeautifulSoup(content, 'lxml')
                text = soup.get_text(separator=' ', strip=True)
                if len(text) > 100:
                    self.page_content[url] = text

            except Exception as e:
                console.print(f"[yellow]Error crawling {url}: {e}[/yellow]")

            progress.update(task, advance=1)

        await context.close()
        return self.document_urls


class StartinUPCrawler(PortalCrawler):
    """Crawler for startinup.up.gov.in"""

    def __init__(self):
        super().__init__("startinup.up.gov.in", "https://startinup.up.gov.in")

        self.pages_to_crawl = [
            "/", "/policy/", "/government-orders/", "/startup-policy-2020/",
            "/startup-policy-2020-first-amendment-2022/", "/it-startup-policy-2017/",
            "/coe/", "/incubators/", "/upcoming-events/", "/recognized-incubators/",
            "/enablers/", "/faqs/",
        ]

        # Known document URLs
        self.known_documents = [
            "/wp-content/uploads/2023/06/User_Manual_Startup_Registration-New.pdf",
            "/wp-content/uploads/2023/06/User_Manual_Incubator_Registration-updated.pdf",
        ]

    async def crawl(self, browser: Browser, progress: Progress) -> Dict[str, DocumentInfo]:
        """Crawl startinup.up.gov.in"""
        task = progress.add_task(f"[cyan]{self.name}", total=len(self.pages_to_crawl) + len(self.known_documents))

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = await context.new_page()

        # Add known documents
        for doc_path in self.known_documents:
            url = f"{self.base_url}{doc_path}"
            self.add_document(url, self.base_url, category="manual")
            progress.update(task, advance=1)

        # Crawl pages
        for page_path in self.pages_to_crawl:
            url = f"{self.base_url}{page_path}"
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(REQUEST_DELAY)

                links = await self.extract_links(page)
                for link in links:
                    if self.is_document_url(link):
                        self.add_document(link, url)
                    # Check for plugin-based GOs
                    elif 'ds-order-manager/uploads' in link:
                        self.add_document(link, url, category="government_order")

                # Extract content
                content = await page.content()
                soup = BeautifulSoup(content, 'lxml')
                main = soup.find('main') or soup.find('article') or soup.body
                if main:
                    text = main.get_text(separator=' ', strip=True)
                    if len(text) > 100:
                        self.page_content[url] = text

            except Exception as e:
                console.print(f"[yellow]Error crawling {url}: {e}[/yellow]")

            progress.update(task, advance=1)

        await context.close()
        return self.document_urls


class UPEIDACrawler(PortalCrawler):
    """Crawler for upeida.up.gov.in"""

    def __init__(self):
        super().__init__("upeida.up.gov.in", "https://upeida.up.gov.in")

        self.pages_to_crawl = [
            "/", "/about-us", "/projects", "/notice-board", "/tenders",
            "/rti", "/gallery", "/contact",
            # Project pages
            "/agra-lucknow-expressway", "/purvanchal-expressway",
            "/bundelkhand-expressway", "/gorakhpur-link-expressway",
            "/ganga-expressway", "/up-defence-industrial-corridor",
        ]

    async def crawl(self, browser: Browser, progress: Progress) -> Dict[str, DocumentInfo]:
        """Crawl upeida.up.gov.in"""
        task = progress.add_task(f"[cyan]{self.name}", total=len(self.pages_to_crawl))

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = await context.new_page()

        for page_path in self.pages_to_crawl:
            url = f"{self.base_url}{page_path}"
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(REQUEST_DELAY)

                links = await self.extract_links(page)
                for link in links:
                    if self.is_document_url(link):
                        self.add_document(link, url)

                content = await page.content()
                soup = BeautifulSoup(content, 'lxml')
                text = soup.get_text(separator=' ', strip=True)
                if len(text) > 100:
                    self.page_content[url] = text

            except Exception as e:
                console.print(f"[yellow]Error crawling {url}: {e}[/yellow]")

            progress.update(task, advance=1)

        await context.close()
        return self.document_urls


class UPSIDACrawler(PortalCrawler):
    """Crawler for UPSIDA (beta.upsidamarketplace.com)"""

    def __init__(self):
        super().__init__("UPSIDA", "http://beta.upsidamarketplace.com")

        self.pages_to_crawl = [
            "/", "/industrial-areas", "/e-services", "/about-us",
        ]

    async def crawl(self, browser: Browser, progress: Progress) -> Dict[str, DocumentInfo]:
        """Crawl UPSIDA marketplace"""
        task = progress.add_task(f"[cyan]{self.name}", total=len(self.pages_to_crawl))

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = await context.new_page()

        for page_path in self.pages_to_crawl:
            url = f"{self.base_url}{page_path}"
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(REQUEST_DELAY)

                links = await self.extract_links(page)
                for link in links:
                    if self.is_document_url(link):
                        self.add_document(link, url)

                content = await page.content()
                soup = BeautifulSoup(content, 'lxml')
                text = soup.get_text(separator=' ', strip=True)
                if len(text) > 100:
                    self.page_content[url] = text

            except Exception as e:
                console.print(f"[yellow]Error crawling {url}: {e}[/yellow]")

            progress.update(task, advance=1)

        await context.close()
        return self.document_urls


class MultiPortalCrawler:
    """Orchestrates crawling across all portals"""

    def __init__(self):
        self.crawlers = [
            InvestUPCrawler(),
            NiveshMitraCrawler(),
            NiveshSarathiCrawler(),
            StartinUPCrawler(),
            UPEIDACrawler(),
            UPSIDACrawler(),
        ]
        self.all_documents: Dict[str, DocumentInfo] = {}
        self.all_content: Dict[str, str] = {}

    async def crawl_all(self) -> Tuple[Dict[str, DocumentInfo], Dict[str, str]]:
        """Crawl all portals"""
        console.print("\n[bold blue]========================================[/bold blue]")
        console.print("[bold blue]  Multi-Portal Crawler for UP Investment[/bold blue]")
        console.print("[bold blue]========================================[/bold blue]\n")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:

                for crawler in self.crawlers:
                    console.print(f"\n[bold cyan]Crawling {crawler.name}...[/bold cyan]")

                    try:
                        docs = await crawler.crawl(browser, progress)
                        self.all_documents.update(docs)
                        self.all_content.update(crawler.page_content)

                        console.print(f"[green]  Found {len(docs)} documents[/green]")
                        console.print(f"[green]  Extracted {len(crawler.page_content)} pages of content[/green]")

                    except Exception as e:
                        console.print(f"[red]Error crawling {crawler.name}: {e}[/red]")

            await browser.close()

        # Save results
        await self.save_results()

        # Print summary
        self.print_summary()

        return self.all_documents, self.all_content

    async def save_results(self):
        """Save crawl results"""
        # Save documents
        docs_file = LOGS_DIR / "all_portal_documents.json"
        docs_data = {url: asdict(doc) for url, doc in self.all_documents.items()}
        async with aiofiles.open(docs_file, 'w') as f:
            await f.write(json.dumps(docs_data, indent=2, ensure_ascii=False))

        # Save content
        content_file = LOGS_DIR / "all_portal_content.json"
        async with aiofiles.open(content_file, 'w') as f:
            await f.write(json.dumps(self.all_content, indent=2, ensure_ascii=False))

        console.print(f"\n[blue]Results saved to {LOGS_DIR}[/blue]")

    def print_summary(self):
        """Print crawl summary"""
        console.print("\n[bold green]========================================[/bold green]")
        console.print("[bold green]           CRAWL SUMMARY[/bold green]")
        console.print("[bold green]========================================[/bold green]\n")

        # Documents by portal
        table = Table(title="Documents by Portal")
        table.add_column("Portal", style="cyan")
        table.add_column("Documents", justify="right", style="green")
        table.add_column("HTML Pages", justify="right", style="yellow")

        portal_docs = {}
        for doc in self.all_documents.values():
            portal_docs[doc.portal] = portal_docs.get(doc.portal, 0) + 1

        portal_pages = {}
        for crawler in self.crawlers:
            portal_pages[crawler.name] = len(crawler.page_content)

        for portal in sorted(portal_docs.keys()):
            table.add_row(portal, str(portal_docs[portal]), str(portal_pages.get(portal, 0)))

        table.add_row("TOTAL", str(len(self.all_documents)), str(len(self.all_content)), style="bold")

        console.print(table)

        # Documents by type
        type_counts = {}
        for doc in self.all_documents.values():
            type_counts[doc.doc_type] = type_counts.get(doc.doc_type, 0) + 1

        console.print("\n[bold]Documents by Type:[/bold]")
        for doc_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            console.print(f"  {doc_type.upper()}: {count}")


async def main():
    """Main entry point"""
    crawler = MultiPortalCrawler()
    await crawler.crawl_all()


if __name__ == "__main__":
    asyncio.run(main())
