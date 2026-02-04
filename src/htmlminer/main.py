import typer
from dotenv import load_dotenv, find_dotenv
import os
import sys
import platform
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import urllib.request
import urllib.error
from typing import Optional
from typing_extensions import Annotated
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.prompt import Prompt, Confirm
import warnings
from urllib.parse import urlparse

# Suppress Pydantic serializer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from .crawler import crawl_domain, fetch_firecrawl_map_urls, fetch_sitemap_urls, scrape_firecrawl_urls
from .graph_state import MODEL_TIERS
from .graph_workflow import run_htmlminer_workflow
from .firecrawl_agent import FirecrawlAgentExtractor, SPARK_MODELS
from .storage import save_results, display_results, save_summary_csv

from .database import init_db, create_session, log_event, save_extractions, save_page_relevance, log_step_timing, get_token_usage_report
from . import __version__

load_dotenv(find_dotenv())

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()

_GITHUB_API_BASE = "https://api.github.com/repos/andreifoldes/htmlminer"
_VERSION_CACHE_PATH = Path("logs") / "version_check.json"
_VERSION_CHECK_INTERVAL = timedelta(hours=24)
STEP_TIMEOUT_S = 600

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

def _parse_version(value: str) -> Optional[tuple[int, int, int]]:
    cleaned = value.strip().lstrip("vV")
    parts = cleaned.split(".")
    if len(parts) < 2:
        return None
    numbers = []
    for part in parts[:3]:
        if not part.isdigit():
            return None
        numbers.append(int(part))
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers)

def _read_version_cache() -> Optional[dict]:
    if not _VERSION_CACHE_PATH.exists():
        return None
    try:
        return json.loads(_VERSION_CACHE_PATH.read_text())
    except Exception:
        return None

def _write_version_cache(latest_version: str) -> None:
    _VERSION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "latest_version": latest_version,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    _VERSION_CACHE_PATH.write_text(json.dumps(payload))

def _fetch_latest_version() -> Optional[str]:
    headers = {"User-Agent": "htmlminer-version-check"}
    release_url = f"{_GITHUB_API_BASE}/releases/latest"
    try:
        req = urllib.request.Request(release_url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.load(response)
            tag = payload.get("tag_name")
            if isinstance(tag, str) and tag.strip():
                return tag.strip()
    except urllib.error.HTTPError as exc:
        if exc.code not in {404, 403}:
            return None
    except Exception:
        return None

    tags_url = f"{_GITHUB_API_BASE}/tags?per_page=1"
    try:
        req = urllib.request.Request(tags_url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.load(response)
            if isinstance(payload, list) and payload:
                tag = payload[0].get("name")
                if isinstance(tag, str) and tag.strip():
                    return tag.strip()
    except Exception:
        return None
    return None

def _maybe_warn_if_outdated() -> None:
    local_version = __version__
    local_parsed = _parse_version(local_version)
    if not local_parsed:
        return

    cached = _read_version_cache()
    latest_version = None
    if cached:
        try:
            checked_at = datetime.fromisoformat(cached.get("checked_at", ""))
        except Exception:
            checked_at = None
        if checked_at and checked_at.tzinfo is None:
            checked_at = checked_at.replace(tzinfo=timezone.utc)
        if checked_at and datetime.now(timezone.utc) - checked_at < _VERSION_CHECK_INTERVAL:
            latest_version = cached.get("latest_version")

    if not latest_version:
        latest_version = _fetch_latest_version()
        if latest_version:
            _write_version_cache(latest_version)

    if not latest_version:
        return

    latest_parsed = _parse_version(latest_version)
    if not latest_parsed:
        return

    if latest_parsed > local_parsed:
        console.print(
            f"[bold yellow]Update available:[/bold yellow] "
            f"you are on v{local_version}, latest is {latest_version}. "
            "Pull the latest changes from GitHub."
        )

def _prompt_for_api_key(key_name: str, description: str, url: str, required: bool = True) -> Optional[str]:
    """
    Prompts the user for an API key and optionally saves it to .env file.

    Args:
        key_name: Name of the environment variable (e.g., "GEMINI_API_KEY")
        description: Human-readable description (e.g., "Gemini API")
        url: URL where users can get the API key
        required: Whether the key is required or optional

    Returns:
        The API key provided by the user, or None if optional and skipped
    """
    console.print()
    if required:
        console.print(f"[bold yellow]⚠ {key_name} is required but not found in .env file[/bold yellow]")
    else:
        console.print(f"[bold blue]ℹ {key_name} not found in .env file[/bold blue]")

    console.print(f"[dim]Get your {description} key from: {url}[/dim]")
    console.print()

    # Ask if user wants to skip (only if optional)
    if not required:
        if not Confirm.ask(f"Do you want to provide {key_name} now?", default=False):
            return None

    # Prompt for the API key
    api_key = Prompt.ask(
        f"[bold]Enter your {description} key[/bold]",
        password=True  # Hide input for security
    )

    if not api_key or not api_key.strip():
        if required:
            console.print(f"[bold red]Error:[/bold red] {key_name} cannot be empty")
            raise typer.Exit(code=1)
        return None

    api_key = api_key.strip()

    # Ask if user wants to save to .env
    console.print()
    save_to_env = Confirm.ask(
        f"[bold]Save {key_name} to .env file?[/bold] (recommended for future runs)",
        default=True
    )

    if save_to_env:
        try:
            env_file = Path(".env")

            # Read existing .env content if it exists
            existing_content = ""
            if env_file.exists():
                existing_content = env_file.read_text()

            # Check if key already exists in file
            lines = existing_content.split('\n')
            key_exists = False
            new_lines = []

            for line in lines:
                if line.strip().startswith(f"{key_name}="):
                    # Replace existing key
                    new_lines.append(f"{key_name}={api_key}")
                    key_exists = True
                else:
                    new_lines.append(line)

            # Add new key if it doesn't exist
            if not key_exists:
                if existing_content and not existing_content.endswith('\n'):
                    new_lines.append('')  # Add newline before new entry
                new_lines.append(f"{key_name}={api_key}")

            # Write back to .env
            env_file.write_text('\n'.join(new_lines))
            console.print(f"[green]✓[/green] {key_name} saved to [bold]{env_file.resolve()}[/bold]")

        except Exception as e:
            console.print(f"[yellow]Warning: Could not save to .env file: {e}[/yellow]")
            console.print("[dim]You can manually add it to .env later[/dim]")
    else:
        console.print(f"[dim]{key_name} will only be used for this session[/dim]")

    # Set the environment variable for current session
    os.environ[key_name] = api_key

    return api_key

@app.callback()
def _main_callback(ctx: typer.Context):
    if getattr(ctx, "resilient_parsing", False):
        return
    _maybe_warn_if_outdated()

@app.command()
def version():
    """Show the version of the application."""
    console.print(f"htmlminer v{__version__}")
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
    llm_timeout: Annotated[int, typer.Option(help="Timeout in seconds for LLM requests (Gemini/DSpy), capped at 600.")] = STEP_TIMEOUT_S,
    gemini_tier: Annotated[str, typer.Option(help="Gemini model tier: 'cheap' or 'expensive'.")] = "cheap",
    smart: Annotated[bool, typer.Option(help="Enable smart crawling to discover and analyze sub-pages (e.g. /about, /research).")] = True,
    limit: Annotated[int, typer.Option(help="Max pages per feature to select from the sitemap when using --smart. Default 10.")] = 10,
    agent: Annotated[bool, typer.Option(help="Use Firecrawl Agent SDK for extraction (requires FIRECRAWL_API_KEY).")] = False,
    spark_model: Annotated[str, typer.Option(help="Spark model for --agent mode: 'mini' (default) or 'pro'.")] = "mini",
    langextract: Annotated[bool, typer.Option(help="Enable LangExtract for intermediate extraction. If disabled (default), full page content is used for synthesis.")] = False,
    langextract_max_char_buffer: Annotated[int, typer.Option(help="Max chars per chunk for LangExtract; smaller values prevent API hangs but increase API calls.")] = 5000,
    gemini_api_key: Annotated[Optional[str], typer.Option(help="Gemini API key (overrides GEMINI_API_KEY env var)")] = None,
    firecrawl_api_key: Annotated[Optional[str], typer.Option(help="Firecrawl API key (overrides FIRECRAWL_API_KEY env var)")] = None,
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
    effective_llm_timeout = llm_timeout if llm_timeout > 0 else STEP_TIMEOUT_S
    if effective_llm_timeout > STEP_TIMEOUT_S:
        effective_llm_timeout = STEP_TIMEOUT_S
    log_event(
        session_id,
        "cli",
        "INFO",
        "Started process command",
        {"file": file, "url": url, "engine": engine, "gemini_tier": gemini_tier, "smart": smart, "limit": limit, "agent": agent, "spark_model": spark_model, "synthesis_top": synthesis_top, "llm_timeout": effective_llm_timeout, "step_timeout": STEP_TIMEOUT_S, "langextract_max_char_buffer": langextract_max_char_buffer},
    )

    mode_label = "Firecrawl Agent" if agent else "Agentic Extraction"
    console.print(Panel.fit(f"[bold cyan]HTMLMiner: {mode_label}[/bold cyan]", border_style="cyan"))
    console.print(f"[dim]Session ID:[/dim] {session_id}")

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

    # Check for Gemini API key (required for all modes except pure firecrawl crawling)
    # Priority: CLI flag > env var > prompt
    api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = _prompt_for_api_key(
            key_name="GEMINI_API_KEY",
            description="Gemini API",
            url="https://aistudio.google.com/app/apikey",
            required=True
        )

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
    
    if llm_timeout < 0:
         msg = "--llm-timeout must be >= 0."
         console.print(f"[bold red]Error:[/bold red] {msg}")
         log_event(session_id, "cli", "ERROR", msg)
         raise typer.Exit(code=1)
    
    if langextract_max_char_buffer < 1:
         msg = "--langextract-max-char-buffer must be >= 1."
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

        # Firecrawl API key is required for agent mode
        # Priority: CLI flag > env var > prompt
        fc_api_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
        if not fc_api_key:
            fc_api_key = _prompt_for_api_key(
                key_name="FIRECRAWL_API_KEY",
                description="Firecrawl API",
                url="https://firecrawl.dev/",
                required=True
            )
        firecrawl_api_key = fc_api_key
    elif engine == "firecrawl":
        # Firecrawl API key is recommended but optional for non-agent crawling
        # Priority: CLI flag > env var > prompt
        fc_api_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
        if not fc_api_key:
            console.print(Panel(
                "[yellow]Note: using 'firecrawl' without FIRECRAWL_API_KEY.[/yellow]\n"
                "For best results, you can provide your Firecrawl API key.",
                border_style="yellow"
            ))
            fc_api_key = _prompt_for_api_key(
                key_name="FIRECRAWL_API_KEY",
                description="Firecrawl API",
                url="https://firecrawl.dev/",
                required=False
            )
        firecrawl_api_key = fc_api_key


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
        extractor = FirecrawlAgentExtractor(
            api_key=firecrawl_api_key,
            spark_model=spark_model,
            session_id=session_id,
            step_timeout_s=STEP_TIMEOUT_S,
        )
        console.print(f"[dim]Using Firecrawl Agent with Spark model: {SPARK_MODELS[spark_model]}[/dim]")
    else:
        # Standard mode: Use LangGraph workflow
        extractor = None  # Graph workflow is invoked per-URL
        model_info = MODEL_TIERS.get(gemini_tier, MODEL_TIERS["cheap"])
        console.print(f"[dim]Using LangGraph workflow with model: {model_info['langchain_model']}[/dim]")

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

        sitemap_by_url = {}
        relevance_scores_by_url = {}  # Track relevance scores across all URLs
        for url_to_process in urls:
            progress.update(overall_task, description=f"[bold]Processing {url_to_process}...")
            log_event(session_id, "cli", "INFO", f"Processing {url_to_process}")
            
            # Sub-task for current URL operations
            current_task = progress.add_task(f"Processing {url_to_process}...", total=None)
            sitemap_urls = []

            try:
                if agent:
                    # Agent mode: Use Firecrawl Agent directly (no crawling/caching needed)
                    progress.update(current_task, description=f"Running Firecrawl Agent on {url_to_process}...")
                    
                    def update_agent_status(msg):
                        if msg.startswith("✓"):
                            progress.console.print(f"  [green]{msg}[/green]")
                        else:
                            progress.update(current_task, description=msg)
                    
                    extraction = extractor.run(
                        url=url_to_process,
                        features=extraction_config or [],
                        status_callback=update_agent_status,
                    )
                    results.append(extraction)
                    sitemap_by_url[url_to_process] = list(sitemap_urls)
                    progress.console.print(f"  [green]✓[/green] Agent extraction complete for {url_to_process}")
                else:
                    # Standard mode: Use LangGraph workflow
                    progress.update(current_task, description=f"Running LangGraph workflow on {url_to_process}...")
                    
                    def update_workflow_status(msg):
                        if msg.startswith("✓"):
                            progress.console.print(f"  [green]{msg}[/green]")
                        else:
                            progress.update(current_task, description=msg, refresh=True)
                    
                    # Run the LangGraph workflow with timeout
                    import concurrent.futures
                    
                    # Define a function to run the workflow
                    def _run_workflow():
                        return run_htmlminer_workflow(
                            url=url_to_process,
                            features=extraction_config or [],
                            api_key=api_key,
                            firecrawl_api_key=firecrawl_api_key,
                            engine=engine,
                            smart_mode=smart,
                            limit=limit,
                            model_tier=gemini_tier,
                            max_paragraphs=max_paragraphs,
                            session_id=session_id,
                            status_callback=update_workflow_status,
                            use_langextract=langextract,
                            langextract_max_char_buffer=langextract_max_char_buffer,
                            synthesis_top=synthesis_top,
                        )

                    # Use ThreadPoolExecutor to enforce timeout
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(_run_workflow)
                        try:
                            # Use step_timeout (10 mins) as the hard limit for the whole process per URL
                            final_state = future.result(timeout=STEP_TIMEOUT_S)
                        except concurrent.futures.TimeoutError:
                            raise TimeoutError(f"Process timed out after {STEP_TIMEOUT_S} seconds")

                    
                    # Extract results from final state
                    extraction = final_state.get("results", {"URL": url_to_process})
                    page_extractions = final_state.get("page_extractions")
                    if page_extractions is not None:
                        # Preserve LangExtract snippets grouped by feature, then page.
                        extraction = dict(extraction)
                        by_feature = {}
                        for page_url, feature_map in page_extractions.items():
                            for feature_name, snippets in feature_map.items():
                                feature_bucket = by_feature.setdefault(feature_name, {})
                                feature_bucket[page_url] = list(snippets)
                        extraction["Raw_Extraction_By_Page"] = by_feature
                    sitemap_urls = final_state.get("sitemap_urls", [])
                    url_relevance_scores = final_state.get("relevance_scores", {})
                    
                    # Track relevance scores for metadata
                    if url_relevance_scores:
                        relevance_scores_by_url[url_to_process] = url_relevance_scores
                    
                    results.append(extraction)
                    sitemap_by_url[url_to_process] = sitemap_urls
                    
                    # Display summary
                    scraped_count = len(final_state.get("scraped_pages", []))
                    progress.console.print(f"  [green]✓[/green] Workflow complete ({scraped_count} pages processed)")

                    # Display any scrape warnings
                    scrape_warnings = final_state.get("scrape_warnings", [])
                    for warning in scrape_warnings:
                        progress.console.print(f"  [bold yellow]⚠[/bold yellow] {warning}")

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
        "sitemaps": sitemap_by_url,
        "relevance_scores": relevance_scores_by_url,
        "parameters": {
            "engine": engine,
            "smart": smart,
            "limit": limit,
            "max_paragraphs": max_paragraphs,
            "gemini_tier": gemini_tier if not agent else None,
            "agent_mode": agent,
            "spark_model": spark_model if agent else None,
            "llm_timeout": effective_llm_timeout,
            "step_timeout": STEP_TIMEOUT_S,
        },
        "urls_processed": len(urls),
        "results_count": len(results),
    }

    save_results(results, output, metadata=metadata)
    save_extractions(results, session_id)
    console.print(f"\n[bold green]✓[/bold green] Results saved to [bold]{os.path.abspath(output)}[/bold]")

    # Export summary CSV
    summary_csv_path = "summary.csv"
    save_summary_csv(results, summary_csv_path, show_full_path=True)

    display_results(results)
    
    if not agent:
        token_report = get_token_usage_report(session_id)
        if token_report.get("steps"):
            token_table = Table(title="Token Usage Report", box=box.SIMPLE)
            token_table.add_column("Step", style="cyan")
            token_table.add_column("Calls", justify="right")
            token_table.add_column("Prompt", justify="right")
            token_table.add_column("Completion", justify="right")
            token_table.add_column("Total", justify="right")
            token_table.add_column("Duration (s)", justify="right")

            for step in token_report["steps"]:
                token_table.add_row(
                    step["step_name"],
                    f"{step['calls']:,}",
                    f"{step['prompt_tokens']:,}",
                    f"{step['completion_tokens']:,}",
                    f"{step['total_tokens']:,}",
                    f"{step['duration_seconds']:.2f}",
                )

            totals = token_report["totals"]
            token_table.add_row(
                "TOTAL",
                f"{totals['calls']:,}",
                f"{totals['prompt_tokens']:,}",
                f"{totals['completion_tokens']:,}",
                f"{totals['total_tokens']:,}",
                f"{totals['duration_seconds']:.2f}",
                style="bold",
            )

            console.print(token_table)
            console.print(f"[dim]Token usage details saved to {os.path.abspath('logs/htmlminer_logs.db')} (step_timings table)[/dim]")
        else:
            console.print("[dim]No token usage data recorded for this session.[/dim]")
    else:
        console.print("[dim]Token usage report not available in agent mode.[/dim]")
    log_event(session_id, "cli", "INFO", "Completed process command", {"results_count": len(results)})

if __name__ == "__main__":
    app()
