import typer
from dotenv import load_dotenv, find_dotenv
import os
from typing import Optional
from typing_extensions import Annotated
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import warnings

# Suppress Pydantic serializer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from .crawler import crawl_and_snapshot
from .agent import AgenticExtractor, MODEL_TIERS
from .storage import save_results, display_results

from .database import init_db, create_session, log_event, save_extractions

load_dotenv(find_dotenv())

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()

@app.command()
def version():
    """Show the version of the application."""
    console.print("htmlminer v0.1.0")

@app.command()

def process(
    file: Annotated[Optional[str], typer.Option(help="Path to markdown file containing URLs")] = None,
    url: Annotated[Optional[str], typer.Option(help="Single URL to process")] = None,
    output: Annotated[str, typer.Option(help="Path to output CSV file")] = "results.csv",
    engine: Annotated[str, typer.Option(help="Engine to use: 'firecrawl' (default) or 'trafilatura'. For Firecrawl, set FIRECRAWL_API_KEY in .env for best results.")] = "firecrawl",
    max_paragraphs: Annotated[int, typer.Option(help="Max paragraphs per dimension in agentic summary")] = 3,
    gemini_tier: Annotated[str, typer.Option(help="Gemini model tier: 'cheap' or 'expensive'.")] = "cheap",
):
    """
    Process URLs from a file, snapshot them, and extract AI risk information.
    """
    init_db()
    session_id = create_session()
    log_event(
        session_id,
        "cli",
        "INFO",
        "Started process command",
        {"file": file, "url": url, "engine": engine, "gemini_tier": gemini_tier},
    )

    console.print(Panel.fit("[bold cyan]HTMLMiner: Agentic Extraction[/bold cyan]", border_style="cyan"))

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

    if engine == "firecrawl" and not os.getenv("FIRECRAWL_API_KEY"):
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
    
    # Initialize Agent
    agent = AgenticExtractor(
        api_key=api_key,
        session_id=session_id,
        extraction_config=extraction_config,
        model_tier=gemini_tier,
    )

    with console.status("[bold green]Processing URLs...[/bold green]", spinner="dots") as status:
        for url in urls:
            status.update(f"Processing [bold]{url}[/bold]...")
            log_event(session_id, "cli", "INFO", f"Processing {url}")
            try:
                # Check for cached snapshot
                from .database import get_latest_snapshot
                from datetime import datetime, timedelta

                snapshot = None
                cached = get_latest_snapshot(url, engine)
                
                if cached:
                    content, timestamp = cached
                    # SQLite returns timestamp as string, parse it
                    try:
                        ts = datetime.fromisoformat(timestamp)
                    except ValueError:
                        ts = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

                    if datetime.now() - ts < timedelta(hours=1):
                         status.stop()
                         should_rescrape = typer.confirm(f"Recent snapshot found for {url} ({timestamp}). Rescrape?", default=False)
                         status.start()
                         
                         if not should_rescrape:
                             snapshot = content
                             log_event(session_id, "cli", "INFO", f"Using cached snapshot for {url}")

                if not snapshot:
                    status.update(f"Snapshotting [bold]{url}[/bold] (engine: {engine})...")
                    snapshot = crawl_and_snapshot(url, engine=engine, session_id=session_id)
                
                status.update(f"Extracting insights from [bold]{url}[/bold]...")
                extraction = agent.extract(snapshot, url, max_paragraphs=max_paragraphs)
                results.append(extraction)
                
            except Exception as e:
                console.print(f"[bold red]Failed to process {url}: {e}[/bold red]")
                log_event(session_id, "cli", "ERROR", f"Failed to process {url}", {"error": str(e)})

    save_results(results, output)
    save_extractions(results, session_id)
    console.print(f"\n[bold green]✓[/bold green] Results saved to [bold]{output}[/bold]")
    display_results(results)
    
    # Token Usage stats
    stats = agent.get_token_usage()
    if stats['total_tokens'] > 0:
        table = Table(title="Token Usage", box=box.ROUNDED)
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="magenta")
        
        table.add_row("Prompt Tokens", f"{stats['prompt_tokens']:,}")
        table.add_row("Completion Tokens", f"{stats['completion_tokens']:,}")
        table.add_row("Total Tokens", f"{stats['total_tokens']:,}")
        
        console.print(table)
    
    log_event(session_id, "cli", "INFO", "Completed process command", {"results_count": len(results), "token_usage": stats})

if __name__ == "__main__":
    app()
