import typer
from dotenv import load_dotenv, find_dotenv
import os
import sys
import platform
from typing import Optional
from typing_extensions import Annotated
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import warnings
from urllib.parse import urlparse

# Suppress Pydantic serializer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from .crawler import crawl_domain, fetch_firecrawl_map_urls, fetch_sitemap_urls, scrape_firecrawl_urls
from .agent import AgenticExtractor, MODEL_TIERS
from .firecrawl_agent import FirecrawlAgentExtractor, SPARK_MODELS
from .storage import save_results, display_results

from .database import init_db, create_session, log_event, save_extractions

load_dotenv(find_dotenv())

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()

def _check_windows_compatibility():
    """
    Performs a compatibility check for Windows systems.
    Validates that required components are available.
    """
    if platform.system() == "Windows":
        # Check Python version
        if sys.version_info < (3, 10):
            console.print("[yellow]Warning: Python 3.10 or higher is recommended on Windows.[/yellow]")

        # Validate that the logs directory can be created
        try:
            logs_dir = "logs"
            os.makedirs(logs_dir, exist_ok=True)
        except Exception as e:
            console.print(f"[red]Error: Cannot create logs directory on Windows: {e}[/red]")
            raise typer.Exit(code=1)

@app.command()
def version():
    """Show the version of the application."""
    console.print("htmlminer v0.1.0")
    console.print(f"Platform: {platform.system()} {platform.release()}")
    console.print(f"Python: {sys.version.split()[0]}")

@app.command()

def process(
    file: Annotated[Optional[str], typer.Option(help="Path to markdown file containing URLs")] = None,
    url: Annotated[Optional[str], typer.Option(help="Single URL to process")] = None,
    output: Annotated[str, typer.Option(help="Path to output file (e.g. results.json or results.csv)")] = "results.json",
    engine: Annotated[str, typer.Option(help="Engine to use: 'firecrawl' (default) or 'trafilatura'. For Firecrawl, set FIRECRAWL_API_KEY in .env for best results.")] = "firecrawl",
    max_paragraphs: Annotated[int, typer.Option(help="Max paragraphs per dimension in agentic summary")] = 3,
    synthesis_top: Annotated[int, typer.Option(help="Max longest snippets per feature to send to synthesis")] = 50,
    gemini_tier: Annotated[str, typer.Option(help="Gemini model tier: 'cheap' or 'expensive'.")] = "cheap",
    smart: Annotated[bool, typer.Option(help="Enable smart crawling to discover and analyze sub-pages (e.g. /about, /research).")] = True,
    limit: Annotated[int, typer.Option(help="Max pages per feature to select from the sitemap when using --smart. Default 10.")] = 10,
    agent: Annotated[bool, typer.Option(help="Use Firecrawl Agent SDK for extraction (requires FIRECRAWL_API_KEY).")] = False,
    spark_model: Annotated[str, typer.Option(help="Spark model for --agent mode: 'mini' (default) or 'pro'.")] = "mini",
):
    """
    Process URLs from a file, snapshot them, and extract AI risk information.
    """
    # Check Windows compatibility
    _check_windows_compatibility()

    # Modify default output filename based on mode
    if output == "results.json" and agent:
        output = "results_agent.json"
    elif output == "results.json" and not agent:
        output = "results.json"  # Keep default
    
    init_db()
    session_id = create_session()
    log_event(
        session_id,
        "cli",
        "INFO",
        "Started process command",
        {"file": file, "url": url, "engine": engine, "gemini_tier": gemini_tier, "smart": smart, "limit": limit, "agent": agent, "spark_model": spark_model, "synthesis_top": synthesis_top},
    )

    mode_label = "Firecrawl Agent" if agent else "Agentic Extraction"
    console.print(Panel.fit(f"[bold cyan]HTMLMiner: {mode_label}[/bold cyan]", border_style="cyan"))

    if not file and not url:
        msg = "At least one of --file or --url must be provided."
        console.print(f"[bold red]Error:[/bold red] {msg}")
        log_event(session_id, "cli", "ERROR", msg)
        raise typer.Exit(code=1)

    urls = []
    if file:
        if not os.path.exists(file):
            msg = f"File {file} not found."
            console.print(f"[bold red]Error:[/bold red] {msg}")
            log_event(session_id, "cli", "ERROR", msg)
            raise typer.Exit(code=1)
        
        with open(file, 'r') as f:
            urls.extend([line.strip() for line in f if line.strip()])

    if url:
        urls.append(url)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
         msg = "Neither GEMINI_API_KEY nor GOOGLE_API_KEY found in environment variables or .env file."
         console.print(f"[bold red]Error:[/bold red] {msg}")
         log_event(session_id, "cli", "ERROR", msg)
         raise typer.Exit(code=1)

    if gemini_tier not in MODEL_TIERS:
         msg = f"Invalid --gemini-tier '{gemini_tier}'. Choose from: {', '.join(MODEL_TIERS.keys())}."
         console.print(f"[bold red]Error:[/bold red] {msg}")
         log_event(session_id, "cli", "ERROR", msg)
         raise typer.Exit(code=1)

    if synthesis_top < 1:
         msg = "--synthesis-top must be >= 1."
         console.print(f"[bold red]Error:[/bold red] {msg}")
         log_event(session_id, "cli", "ERROR", msg)
         raise typer.Exit(code=1)

    # Validate spark model for agent mode
    if agent:
        if spark_model not in SPARK_MODELS:
            msg = f"Invalid --spark-model '{spark_model}'. Choose from: {', '.join(SPARK_MODELS.keys())}."
            console.print(f"[bold red]Error:[/bold red] {msg}")
            log_event(session_id, "cli", "ERROR", msg)
            raise typer.Exit(code=1)
        
        firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        if not firecrawl_api_key:
            msg = "FIRECRAWL_API_KEY is required for --agent mode. Set it in your .env file."
            console.print(f"[bold red]Error:[/bold red] {msg}")
            log_event(session_id, "cli", "ERROR", msg)
            raise typer.Exit(code=1)
    elif engine == "firecrawl" and not os.getenv("FIRECRAWL_API_KEY"):
        console.print(Panel("[yellow]Note: using 'firecrawl' without FIRECRAWL_API_KEY.[/yellow]\nFor best results, create an account at firecrawl.dev and add your API key to .env.", border_style="yellow"))


    import json
    
    config_path = "config.json"
    extraction_config = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            try:
                extraction_config = json.load(f).get("features", [])
            except json.JSONDecodeError:
                console.print(f"[bold red]Error:[/bold red] Failed to parse {config_path}.")
                raise typer.Exit(code=1)
    
    results = []
    session_page_cache = {}
    
    # Initialize the appropriate extractor
    if agent:
        firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        extractor = FirecrawlAgentExtractor(
            api_key=firecrawl_api_key,
            spark_model=spark_model,
            session_id=session_id,
        )
        console.print(f"[dim]Using Firecrawl Agent with Spark model: {SPARK_MODELS[spark_model]}[/dim]")
    else:
        extractor = AgenticExtractor(
            api_key=api_key,
            session_id=session_id,
            extraction_config=extraction_config,
            model_tier=gemini_tier,
            synthesis_top=synthesis_top,
        )

    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        
        overall_task = progress.add_task(f"[bold green]Processing {len(urls)} URLs...", total=len(urls))

        for url_to_process in urls:
            progress.update(overall_task, description=f"[bold]Processing {url_to_process}...")
            log_event(session_id, "cli", "INFO", f"Processing {url_to_process}")
            
            # Sub-task for current URL operations
            current_task = progress.add_task(f"Processing {url_to_process}...", total=None)

            try:
                if agent:
                    # Agent mode: Use Firecrawl Agent directly (no crawling/caching needed)
                    progress.update(current_task, description=f"Running Firecrawl Agent on {url_to_process}...")
                    
                    def update_agent_status(msg):
                        progress.update(current_task, description=msg)
                    
                    extraction = extractor.run(
                        url=url_to_process,
                        features=extraction_config or [],
                        status_callback=update_agent_status,
                    )
                    results.append(extraction)
                    progress.console.print(f"  [green]✓[/green] Agent extraction complete for {url_to_process}")
                else:
                    # Standard mode: Crawl + Extract
                    # Check for cached snapshot
                    from .database import get_latest_snapshot
                    from datetime import datetime, timedelta

                    snapshot = None
                    crawl_results = []
                    cached = get_latest_snapshot(url_to_process, engine)
                    
                    if cached:
                        content, timestamp = cached
                        # SQLite returns timestamp as string, parse it
                        try:
                            ts = datetime.fromisoformat(timestamp)
                        except ValueError:
                            ts = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

                        if datetime.now() - ts < timedelta(hours=1):
                            progress.update(current_task, visible=False)
                            progress.stop()
                            should_rescrape = typer.confirm(f"Recent snapshot found for {url_to_process} ({timestamp}). Rescrape?", default=False)
                            progress.start()
                            
                            if not should_rescrape:
                                snapshot = content
                                log_event(session_id, "cli", "INFO", f"Using cached snapshot for {url_to_process}")

                    sitemap_urls = []
                    selected_by_feature = {}
                    if not snapshot:
                        crawl_limit = limit if smart else 1

                        def update_crawl_progress(completed, total):
                            progress.update(current_task, completed=completed, total=total)
                        
                        def update_crawl_status(msg):
                            progress.update(current_task, description=msg)

                        if engine == "firecrawl":
                            firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
                            if not firecrawl_api_key:
                                raise Exception("FIRECRAWL_API_KEY environment variable not set.")

                            progress.update(current_task, description=f"Fetching sitemap for {url_to_process}...", total=None)
                            sitemap_urls = fetch_firecrawl_map_urls(
                                url_to_process,
                                firecrawl_api_key,
                                limit=5000,
                                include_subdomains=False,
                                sitemap_mode="include",
                            )
                            if not sitemap_urls:
                                sitemap_urls = fetch_sitemap_urls(url_to_process)

                            feature_page_limit = limit
                            selection = extractor.select_top_pages_for_features(
                                url_to_process,
                                sitemap_urls,
                                feature_page_limit,
                                status_callback=update_crawl_status,
                            )
                            selected_urls = selection.get("all", [])
                            selected_by_feature = selection.get("by_feature", {})

                            cached_results = []
                            urls_to_scrape = []
                            for page_url in selected_urls:
                                if page_url in session_page_cache:
                                    cached_results.append(
                                        (page_url, session_page_cache[page_url])
                                    )
                                else:
                                    urls_to_scrape.append(page_url)

                            progress.update(
                                current_task,
                                description=f"Scraping {len(urls_to_scrape)} selected page{'s' if len(urls_to_scrape) != 1 else ''}...",
                                visible=True,
                                total=len(urls_to_scrape) if urls_to_scrape else None,
                            )

                            crawl_results = []
                            if urls_to_scrape:
                                crawl_results = scrape_firecrawl_urls(
                                    urls_to_scrape,
                                    firecrawl_api_key,
                                    session_id=session_id,
                                    status_callback=update_crawl_status,
                                    progress_callback=update_crawl_progress,
                                )
                            if cached_results:
                                crawl_results = cached_results + crawl_results
                            for page_url, content in crawl_results:
                                session_page_cache[page_url] = content
                        else:
                            progress.update(
                                current_task,
                                description=f"Snapshotting {url_to_process} (engine: {engine}, limit: {crawl_limit})...",
                                visible=True,
                                total=crawl_limit if crawl_limit > 1 else None,
                            )

                            # crawl_domain returns list of (url, content)
                            crawl_results = crawl_domain(
                                url_to_process,
                                engine=engine,
                                limit=crawl_limit,
                                session_id=session_id,
                                status_callback=update_crawl_status,
                                progress_callback=update_crawl_progress,
                            )
                        
                        if not crawl_results:
                            raise Exception("No content retrieved.")
                    
                        progress.console.print(f"  [green]✓[/green] Snapshot captured ({len(crawl_results)} page{'s' if len(crawl_results)>1 else ''})")
                        
                        # Display scraped URLs
                        if len(crawl_results) > 1:
                            from rich.tree import Tree
                            tree = Tree(f"[dim]Scraped paths for {url_to_process}:[/dim]")
                            for r_url, _ in crawl_results:
                                tree.add(f"[dim]{r_url}[/dim]")
                            progress.console.print(tree)
                    else:
                        # Using cached snapshot
                        progress.console.print(f"  [green]✓[/green] Using cached snapshot")
                    
                    if not sitemap_urls:
                        progress.update(current_task, description=f"Fetching sitemap for {url_to_process}...", total=None)
                        firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
                        if engine == "firecrawl" and firecrawl_api_key:
                            sitemap_urls = fetch_firecrawl_map_urls(
                                url_to_process,
                                firecrawl_api_key,
                                limit=5000,
                                include_subdomains=False,
                                sitemap_mode="include",
                            )
                        if not sitemap_urls:
                            sitemap_urls = fetch_sitemap_urls(url_to_process)

                    scraped_urls = [page_url for page_url, _ in crawl_results] if crawl_results else [url_to_process]
                    scraped_paths = []
                    seen_paths = set()
                    for scraped_url in scraped_urls:
                        parsed = urlparse(scraped_url)
                        path = parsed.path or "/"
                        if parsed.query:
                            path = f"{path}?{parsed.query}"
                        if path not in seen_paths:
                            scraped_paths.append(path)
                            seen_paths.add(path)

                    site_context = {
                        "sitemap_urls": sitemap_urls,
                        "scraped_paths": scraped_paths,
                        "scraped_urls": scraped_urls,
                        "selected_urls_by_feature": selected_by_feature if engine == "firecrawl" else {},
                        "max_items": 50,
                    }

                    progress.update(current_task, description=f"Extracting insights from {url_to_process}...", total=None) 
                    
                    def update_extract_status(msg):
                        progress.update(current_task, description=msg)
                    
                    # Extraction
                    if crawl_results:
                        extraction = extractor.extract_from_pages(
                            crawl_results,
                            url_to_process,
                            max_paragraphs=max_paragraphs,
                            status_callback=update_extract_status,
                            site_context=site_context,
                        )
                    else:
                        extraction = extractor.extract(
                            snapshot,
                            url_to_process,
                            max_paragraphs=max_paragraphs,
                            status_callback=update_extract_status,
                            site_context=site_context,
                        )
                    results.append(extraction)
                
                progress.remove_task(current_task)
                progress.advance(overall_task)
                
            except Exception as e:
                progress.console.print(f"[bold red]Failed to process {url_to_process}: {e}[/bold red]")
                log_event(session_id, "cli", "ERROR", f"Failed to process {url_to_process}", {"error": str(e)})
                progress.remove_task(current_task) # cleanup

    # Build metadata for JSON output
    from datetime import datetime
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "parameters": {
            "engine": engine,
            "smart": smart,
            "limit": limit,
            "max_paragraphs": max_paragraphs,
            "gemini_tier": gemini_tier if not agent else None,
            "agent_mode": agent,
            "spark_model": spark_model if agent else None,
        },
        "urls_processed": len(urls),
        "results_count": len(results),
    }

    save_results(results, output, metadata=metadata)
    save_extractions(results, session_id)
    console.print(f"\n[bold green]✓[/bold green] Results saved to [bold]{output}[/bold]")
    display_results(results)
    
    # Token Usage stats (only for standard extraction mode)
    if not agent and hasattr(extractor, 'get_token_usage'):
        stats = extractor.get_token_usage()
        if stats['total_tokens'] > 0:
            table = Table(title="Token Usage", box=box.ROUNDED)
            table.add_column("Type", style="cyan")
            table.add_column("Count", style="magenta")
            
            table.add_row("Prompt Tokens", f"{stats['prompt_tokens']:,}")
            table.add_row("Completion Tokens", f"{stats['completion_tokens']:,}")
            table.add_row("Total Tokens", f"{stats['total_tokens']:,}")
            
            console.print(table)
        log_event(session_id, "cli", "INFO", "Completed process command", {"results_count": len(results), "token_usage": stats})
    else:
        log_event(session_id, "cli", "INFO", "Completed process command", {"results_count": len(results)})

if __name__ == "__main__":
    app()
