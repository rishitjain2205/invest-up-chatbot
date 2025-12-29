"""
Shasanadesh Semi-Automated Crawler
Crawls historical Government Orders from UP Shasanadesh portal
Requires manual CAPTCHA solving once, then automates everything else
"""
import asyncio
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs
from dataclasses import dataclass, field, asdict
from typing import Set, Dict, List, Optional
from datetime import datetime

from playwright.async_api import async_playwright, Page, Browser
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
import aiohttp
import aiofiles

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DIR, LOGS_DIR

console = Console()


@dataclass
class ShasanadeshGO:
    """Government Order from Shasanadesh"""
    url: str
    title: str
    go_number: str
    go_date: str
    department: str
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    downloaded: bool = False
    local_path: Optional[str] = None


class ShasanadeshCrawler:
    """
    Semi-automated crawler for Shasanadesh

    Flow:
    1. Opens browser with search form
    2. Selects Industrial Development Department
    3. Sets date range (2015-2025)
    4. USER solves CAPTCHA manually
    5. Script clicks search and paginates through all results
    6. Extracts all PDF links
    """

    BASE_URL = "https://shasanadesh.up.gov.in"

    # Industrial Development Department value in dropdown
    INDUSTRIAL_DEV_DEPT = "औद्योगिक विकास विभाग"

    def __init__(self):
        self.discovered_gos: Dict[str, ShasanadeshGO] = {}
        self.total_pages = 0

    async def run_interactive_crawl(self):
        """Run the interactive crawl with manual CAPTCHA"""
        console.print(Panel(
            "[bold]Shasanadesh Historical GO Crawler[/bold]\n\n"
            "This will crawl ~1500 Government Orders from Industrial Development Dept\n"
            "Date Range: 2015-2025\n\n"
            "[yellow]You will need to solve ONE CAPTCHA manually[/yellow]",
            title="Starting Shasanadesh Crawl"
        ))

        async with async_playwright() as p:
            # Launch browser in headed mode so user can see and interact
            browser = await p.chromium.launch(
                headless=False,  # Show browser for CAPTCHA
                slow_mo=100  # Slow down for visibility
            )

            context = await browser.new_context(
                viewport={"width": 1280, "height": 800},
                ignore_https_errors=True
            )
            page = await context.new_page()

            try:
                # Step 1: Navigate to Shasanadesh
                console.print("\n[cyan]Step 1: Opening Shasanadesh...[/cyan]")
                await page.goto(self.BASE_URL, wait_until="networkidle", timeout=60000)
                await asyncio.sleep(2)

                # Step 2: Select Industrial Development Department
                console.print("[cyan]Step 2: Selecting Industrial Development Department...[/cyan]")

                # Wait for dropdown to be ready
                await page.wait_for_selector("#ddldept", timeout=10000)

                # Select department by visible text
                await page.select_option("#ddldept", label=self.INDUSTRIAL_DEV_DEPT)
                await asyncio.sleep(1)

                # Step 3: Skip date range (leave blank to get all records)
                console.print("[cyan]Step 3: Skipping date filter (will get all records)...[/cyan]")
                # Date filters are optional - leaving blank to get all GOs
                await asyncio.sleep(1)

                # Step 4: Wait for user to solve CAPTCHA
                console.print("\n" + "="*60)
                console.print("[bold yellow]STEP 4: PLEASE SOLVE THE CAPTCHA IN THE BROWSER[/bold yellow]")
                console.print("="*60)
                console.print("\n[yellow]1. Look at the CAPTCHA image in the browser[/yellow]")
                console.print("[yellow]2. Type the numbers you see in the CAPTCHA box[/yellow]")
                console.print("[yellow]3. Click the SEARCH button (खोजें)[/yellow]")
                console.print("\n[bold green]You have 2 minutes to solve the CAPTCHA...[/bold green]")
                console.print("[cyan]The script will automatically continue once results appear.[/cyan]")

                # Wait for user to solve CAPTCHA and results to load (up to 2 minutes)
                try:
                    # Wait for results table to appear (indicates search was successful)
                    await page.wait_for_selector("table#grdmain, table.table, #grdmain", timeout=120000)
                    console.print("\n[green]Search results detected! Continuing...[/green]")
                except:
                    console.print("\n[yellow]Timeout waiting for results. Checking page state...[/yellow]")

                await asyncio.sleep(3)

                # Step 5: Check if search results loaded
                # Look for results table or pagination
                try:
                    tables = await page.query_selector_all("table")
                    if not tables:
                        console.print("[yellow]No results table found. Waiting 30 more seconds...[/yellow]")
                        await asyncio.sleep(30)
                except Exception as check_err:
                    console.print(f"[yellow]Check error: {str(check_err)}[/yellow]")

                # Step 6: Extract results from current page and paginate
                console.print("\n[cyan]Step 5: Extracting Government Orders...[/cyan]")

                page_num = 1
                while True:
                    console.print(f"\n[blue]Processing page {page_num}...[/blue]")

                    # Extract GOs from current page
                    gos_on_page = await self.extract_gos_from_page(page)
                    console.print(f"[green]Found {len(gos_on_page)} GOs on page {page_num}[/green]")

                    for go in gos_on_page:
                        if go.url not in self.discovered_gos:
                            self.discovered_gos[go.url] = go

                    # Save progress
                    await self.save_results()

                    # Check for next page
                    next_button = await page.query_selector("a:has-text('Next')")
                    if not next_button:
                        # Try Hindi text
                        next_button = await page.query_selector("a:has-text('अगला')")

                    if not next_button:
                        # Try finding pagination links
                        next_page = await page.query_selector(f"a:has-text('{page_num + 1}')")
                        if next_page:
                            next_button = next_page

                    if next_button:
                        is_disabled = await next_button.get_attribute("disabled")
                        if is_disabled:
                            console.print("[yellow]Reached last page[/yellow]")
                            break

                        try:
                            # Use JavaScript click to avoid stability issues
                            await next_button.evaluate("el => el.click()")
                            await asyncio.sleep(3)
                            try:
                                await page.wait_for_load_state("networkidle", timeout=10000)
                            except:
                                pass  # Continue even if timeout
                            page_num += 1
                        except Exception as click_err:
                            console.print(f"[yellow]Pagination error: {str(click_err)[:50]}[/yellow]")
                            # Try alternative pagination method
                            try:
                                await page.evaluate(f"document.querySelector('a:contains({page_num + 1})').click()")
                                await asyncio.sleep(3)
                                page_num += 1
                            except:
                                console.print("[yellow]Could not navigate to next page[/yellow]")
                                break
                    else:
                        console.print("[yellow]No more pages found[/yellow]")
                        break

                    # Safety limit
                    if page_num > 200:
                        console.print("[yellow]Reached page limit (200)[/yellow]")
                        break

                self.total_pages = page_num

            except Exception as e:
                console.print(f"[red]Error during crawl: {str(e)}[/red]")
                import traceback
                traceback.print_exc()

            finally:
                # Save final results
                await self.save_results()

                console.print("\n[bold green]Crawl Complete![/bold green]")
                console.print(f"Total GOs discovered: {len(self.discovered_gos)}")
                console.print(f"Total pages processed: {self.total_pages}")

                console.print("\n[yellow]Closing browser in 5 seconds...[/yellow]")
                await asyncio.sleep(5)
                await browser.close()

        return self.discovered_gos

    async def extract_gos_from_page(self, page: Page) -> List[ShasanadeshGO]:
        """Extract GO information from current results page"""
        gos = []

        try:
            content = await page.content()
            soup = BeautifulSoup(content, 'lxml')

            # Find all rows in results table
            # Shasanadesh typically shows results in a table with PDF links

            # Look for PDF links
            pdf_links = soup.find_all('a', href=re.compile(r'frmPDF\.aspx|\.pdf', re.I))

            for link in pdf_links:
                href = link.get('href', '')
                if not href:
                    continue

                # Build full URL
                if href.startswith('/'):
                    full_url = f"{self.BASE_URL}{href}"
                elif not href.startswith('http'):
                    full_url = f"{self.BASE_URL}/{href}"
                else:
                    full_url = href

                # Extract title/text
                title = link.get_text(strip=True) or "Untitled GO"

                # Try to find GO number and date from parent row
                parent_row = link.find_parent('tr')
                go_number = ""
                go_date = ""

                if parent_row:
                    cells = parent_row.find_all('td')
                    for cell in cells:
                        text = cell.get_text(strip=True)
                        # Look for date pattern (dd/mm/yyyy)
                        date_match = re.search(r'\d{2}/\d{2}/\d{4}', text)
                        if date_match:
                            go_date = date_match.group()
                        # Look for GO number pattern
                        if 'GO' in text.upper() or re.search(r'\d+/\d+', text):
                            go_number = text

                go = ShasanadeshGO(
                    url=full_url,
                    title=title[:200],  # Limit title length
                    go_number=go_number,
                    go_date=go_date,
                    department=self.INDUSTRIAL_DEV_DEPT
                )
                gos.append(go)

            # Also look for direct table rows with GO info
            rows = soup.find_all('tr')
            for row in rows:
                links = row.find_all('a', href=True)
                for link in links:
                    href = link['href']
                    if 'pdf' in href.lower() or 'frmPDF' in href:
                        # Already processed above
                        continue
                    if 'view' in href.lower() or 'download' in href.lower():
                        full_url = urljoin(self.BASE_URL, href)
                        if full_url not in [g.url for g in gos]:
                            gos.append(ShasanadeshGO(
                                url=full_url,
                                title=link.get_text(strip=True)[:200],
                                go_number="",
                                go_date="",
                                department=self.INDUSTRIAL_DEV_DEPT
                            ))

        except Exception as e:
            console.print(f"[yellow]Error extracting GOs: {e}[/yellow]")

        return gos

    async def save_results(self):
        """Save discovered GOs to JSON"""
        output_file = LOGS_DIR / "shasanadesh_gos.json"

        data = {url: asdict(go) for url, go in self.discovered_gos.items()}

        async with aiofiles.open(output_file, 'w') as f:
            await f.write(json.dumps(data, indent=2, ensure_ascii=False))

        console.print(f"[blue]Progress saved: {len(self.discovered_gos)} GOs[/blue]")


class ShasanadeshDownloader:
    """Download PDFs from discovered Shasanadesh GOs"""

    def __init__(self):
        self.stats = {"success": 0, "failed": 0}

    async def download_all(self):
        """Download all discovered GOs"""
        gos_file = LOGS_DIR / "shasanadesh_gos.json"

        if not gos_file.exists():
            console.print("[red]No Shasanadesh GOs found. Run crawler first.[/red]")
            return

        async with aiofiles.open(gos_file, 'r') as f:
            content = await f.read()
            all_gos = json.loads(content)

        console.print(f"\n[bold blue]Downloading {len(all_gos)} Shasanadesh GOs...[/bold blue]")

        # Create output directory
        output_dir = RAW_DIR / "shasanadesh_up_gov_in" / "pdf"
        output_dir.mkdir(parents=True, exist_ok=True)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        connector = aiohttp.TCPConnector(limit=5, ssl=False)
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Downloading...", total=len(all_gos))

                for url, go_data in all_gos.items():
                    if go_data.get("downloaded"):
                        progress.update(task, advance=1)
                        continue

                    try:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                            if response.status == 200:
                                content = await response.read()

                                # Generate filename
                                filename = f"go_{hash(url) % 100000:05d}.pdf"
                                file_path = output_dir / filename

                                async with aiofiles.open(file_path, 'wb') as f:
                                    await f.write(content)

                                go_data["downloaded"] = True
                                go_data["local_path"] = str(file_path)
                                self.stats["success"] += 1
                            else:
                                self.stats["failed"] += 1

                    except Exception as e:
                        self.stats["failed"] += 1

                    progress.update(task, advance=1)

        # Save updated results
        async with aiofiles.open(gos_file, 'w') as f:
            await f.write(json.dumps(all_gos, indent=2, ensure_ascii=False))

        console.print(f"\n[bold green]Download Summary:[/bold green]")
        console.print(f"  Successful: {self.stats['success']}")
        console.print(f"  Failed: {self.stats['failed']}")


async def main():
    """Main entry point"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "download":
        downloader = ShasanadeshDownloader()
        await downloader.download_all()
    else:
        crawler = ShasanadeshCrawler()
        await crawler.run_interactive_crawl()


if __name__ == "__main__":
    asyncio.run(main())
