"""
HTMLMiner Graph Nodes

Individual node implementations for the LangGraph workflow.
Each node takes the workflow state and returns updates to it.
"""

import time
import datetime
import json
import re
from typing import Optional
from urllib.parse import urlparse

import langextract as lx
from langextract.core.data import ExampleData, Extraction
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from rich.console import Console
from rich.prompt import Confirm

from .graph_state import HTMLMinerState, MODEL_TIERS
from .graph_callbacks import StepTimingWrapper, TimingTokenCallback
from .crawler import (
    fetch_sitemap_urls,
    fetch_firecrawl_map_urls,
    scrape_firecrawl_urls,
    crawl_domain,
)
from .database import log_event, save_page_relevance, get_latest_snapshot
from .formatting import (
    is_list_output_format,
    list_output_instruction,
    normalize_bullet_list,
)

console = Console()

LANGEXTRACT_EXTRACTION_PASSES = 1
LANGEXTRACT_MAX_CHAR_BUFFER = 50000  # 50KB - larger buffer = fewer API calls = faster extraction
CACHE_TTL_SECONDS = 60 * 60


def _parse_snapshot_timestamp(timestamp: str) -> Optional[datetime.datetime]:
    if not timestamp:
        return None
    try:
        return datetime.datetime.fromisoformat(timestamp)
    except ValueError:
        return None


def _get_cached_snapshot(
    page_url: str,
    engine: str,
    session_id: str,
    status_callback: Optional[callable],
) -> Optional[str]:
    snapshot = get_latest_snapshot(page_url, engine)
    if not snapshot:
        return None
    content, timestamp = snapshot
    if not content:
        return None
    snapshot_time = _parse_snapshot_timestamp(timestamp)
    if not snapshot_time:
        return None
    age_seconds = (datetime.datetime.now() - snapshot_time).total_seconds()
    if age_seconds > CACHE_TTL_SECONDS:
        return None
    age_minutes = max(1, int(age_seconds // 60))
    # Automatically use cache if within TTL, don't ask
    # Default behavior is to use cache
    if status_callback:
        status_callback(f"Using cached snapshot for {page_url} ({age_minutes}m old).")
    if session_id:
        log_event(
            session_id,
            "graph",
            "INFO",
            "Using cached snapshot for page",
            {"url": page_url, "age_minutes": age_minutes},
        )
    return content


# =============================================================================
# Pydantic Schemas for Structured Output
# =============================================================================

class PageSelection(BaseModel):
    """Schema for LLM page selection output."""
    url: str = Field(description="The URL of the selected page")
    relevance_score: int = Field(
        description="Overall relevance score from 1-10, where 10 is most relevant",
        ge=1,
        le=10
    )
    feature_scores: dict[str, int] = Field(
        description="Per-feature relevance scores (1-10) e.g. {'Risk': 8, 'Goal': 9, 'Method': 7}",
        default_factory=dict
    )


class PageSelections(BaseModel):
    """List of selected pages with scores."""
    selections: list[PageSelection] = Field(
        description="List of selected pages ordered by relevance score descending"
    )


class FeatureSynthesis(BaseModel):
    """Schema for feature synthesis output."""
    summary: str = Field(description="Synthesized summary of the feature")


# =============================================================================
# Node: Check Engine (Router)
# =============================================================================

def check_engine_node(state: HTMLMinerState) -> dict:
    """
    Router node: determines the crawling strategy based on engine and smart mode.
    
    Returns routing decision in state for conditional edge.
    """
    engine = state.get("engine", "firecrawl")
    smart_mode = state.get("smart_mode", True)
    
    if state.get("status_callback"):
        state["status_callback"](f"Checking engine: {engine}, smart={smart_mode}")
    
    # Log the decision
    log_event(
        session_id=state.get("session_id", ""),
        component="graph",
        level="INFO",
        message=f"Engine check: {engine}, smart_mode={smart_mode}",
    )
    
    return {"engine": engine, "smart_mode": smart_mode}


def route_by_engine(state: HTMLMinerState) -> str:
    """
    Conditional edge function: routes to smart or simple crawling.
    """
    engine = state.get("engine", "firecrawl")
    smart_mode = state.get("smart_mode", True)
    
    if engine == "firecrawl" and smart_mode:
        return "smart"
    return "simple"


# =============================================================================
# Node: Fetch Sitemap
# =============================================================================

def fetch_sitemap_node(state: HTMLMinerState) -> dict:
    """
    Fetches sitemap URLs using Firecrawl's map API or traditional sitemap parsing.
    """
    url = state["url"]
    session_id = state.get("session_id", "")
    firecrawl_api_key = state.get("firecrawl_api_key")
    
    if state.get("status_callback"):
        state["status_callback"]("Fetching sitemap...")
    
    with StepTimingWrapper("fetch_sitemap", session_id, url, status_callback=state.get("status_callback")) as timer:
        sitemap_urls = []
        
        # Try Firecrawl map API first if we have a key
        if firecrawl_api_key:
            try:
                sitemap_urls = fetch_firecrawl_map_urls(
                    site_url=url,
                    api_key=firecrawl_api_key,
                    limit=5000,
                )
            except Exception as e:
                log_event(session_id, "graph", "WARNING", f"Firecrawl map failed: {e}")
        
        # Fallback to traditional sitemap
        if not sitemap_urls:
            try:
                sitemap_urls = fetch_sitemap_urls(url, max_urls=500)
            except Exception as e:
                log_event(session_id, "graph", "WARNING", f"Sitemap fetch failed: {e}")
        
        timer.set_details({"url_count": len(sitemap_urls)})
    
    log_event(session_id, "graph", "INFO", f"Fetched {len(sitemap_urls)} sitemap URLs")
    
    return {"sitemap_urls": sitemap_urls}


# =============================================================================
# Node: Filter URLs
# =============================================================================

# Extensions to skip
DISALLOWED_EXTENSIONS = frozenset([
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".zip", ".gz", ".tar",
    ".css", ".js", ".json", ".xml", ".rss", ".atom",
])


def filter_urls_node(state: HTMLMinerState) -> dict:
    """
    Filters candidate URLs by domain and file extension.
    """
    url = state["url"]
    sitemap_urls = state.get("sitemap_urls", [])
    session_id = state.get("session_id", "")
    
    if state.get("status_callback"):
        state["status_callback"](f"Filtering {len(sitemap_urls)} URLs...")
    
    with StepTimingWrapper("filter_urls", session_id, url, status_callback=state.get("status_callback")) as timer:
        base_domain = urlparse(url).netloc.lower()
        
        def is_same_domain(u: str) -> bool:
            parsed = urlparse(u)
            candidate_domain = parsed.netloc.lower()
            return (
                candidate_domain == base_domain
                or candidate_domain.endswith("." + base_domain)
            )
        
        def is_allowed(u: str) -> bool:
            path = urlparse(u).path.lower()
            return not any(path.endswith(ext) for ext in DISALLOWED_EXTENSIONS)
        
        filtered = [u for u in sitemap_urls if is_same_domain(u) and is_allowed(u)]
        
        # Limit to reasonable number for LLM selection
        max_candidates = 200
        if len(filtered) > max_candidates:
            filtered = filtered[:max_candidates]
        
        timer.set_details({
            "input_count": len(sitemap_urls),
            "output_count": len(filtered),
        })
    
    log_event(session_id, "graph", "INFO", f"Filtered to {len(filtered)} candidate URLs")
    
    return {"filtered_urls": filtered}


# =============================================================================
# Node: Select Pages (LLM-based)
# =============================================================================

def select_pages_node(state: HTMLMinerState) -> dict:
    """
    Uses LLM to select the most relevant pages for feature extraction.
    Replaces the DSPy PageSelector signature.
    """
    url = state["url"]
    filtered_urls = state.get("filtered_urls", [])
    limit = state.get("limit", 10)
    features = state.get("features", [])
    api_key = state.get("api_key", "")
    model_tier = state.get("model_tier", "cheap")
    session_id = state.get("session_id", "")
    
    if not filtered_urls:
        return {"selected_pages": [], "relevance_scores": {}}
    
    if state.get("status_callback"):
        state["status_callback"](f"Selecting top {limit} pages from {len(filtered_urls)} candidates...")
    
    model_name = MODEL_TIERS.get(model_tier, MODEL_TIERS["cheap"])["langchain_model"]
    
    with StepTimingWrapper("select_pages", session_id, url, status_callback=state.get("status_callback")) as timer:
        # Create callback for token tracking
        callback = TimingTokenCallback(session_id, "select_pages_llm")
        
        # Initialize LLM with structured output
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
        )
        structured_llm = llm.with_structured_output(PageSelections)
        
        # Build feature descriptions
        feature_desc = "\n".join([
            f"- {f['name']}: {f['description']}"
            for f in features
        ])
        feature_names = [f['name'] for f in features]
        
        # Build URL list
        url_list = "\n".join([f"- {u}" for u in filtered_urls[:100]])  # Cap at 100 for context limit
        
        prompt = f"""You are analyzing the website {url} to find pages relevant to these features:

{feature_desc}

Select the top {limit} most relevant pages from this list of URLs. 
For each page, assign:
1. An overall relevance score (1-10)
2. A per-feature relevance score for each feature: {feature_names}

URLs:
{url_list}

Select pages that are most likely to contain information about the features above."""

        try:
            result = structured_llm.invoke(
                [HumanMessage(content=prompt)],
                config={"callbacks": [callback]},
            )
            
            selected = [
                {"url": s.url, "relevance_score": s.relevance_score}
                for s in result.selections[:limit]
            ]
            
            # Build relevance scores dict for database (with per-feature scores)
            relevance_scores = {}
            for s in result.selections[:limit]:
                scores = {"overall": s.relevance_score}
                scores.update(s.feature_scores)
                relevance_scores[s.url] = scores
            
            timer.set_details({
                "selected_count": len(selected),
                "tokens": callback.get_totals(),
            })
            
        except Exception as e:
            log_event(session_id, "graph", "ERROR", f"LLM selection failed: {e}")
            # Fallback to heuristic selection
            selected = _heuristic_select(filtered_urls, features, limit)
            # Build fallback scores with per-feature defaults
            relevance_scores = {}
            for s in selected:
                scores = {"overall": 5}
                for f in features:
                    scores[f["name"]] = 5
                relevance_scores[s["url"]] = scores
            timer.set_details({"fallback": True, "error": str(e)})
    
    # Save relevance scores to database
    if relevance_scores and session_id:
        if state.get("status_callback"):
            state["status_callback"]("Saving page relevance scores...")
        save_page_relevance(session_id, url, relevance_scores)
    
    return {
        "selected_pages": selected,
        "relevance_scores": relevance_scores,
    }


def _heuristic_select(urls: list[str], features: list[dict], limit: int) -> list[dict]:
    """Fallback heuristic selection based on URL keywords."""
    keywords = set()
    for f in features:
        name = f.get("name", "").lower()
        desc = f.get("description", "").lower()
        keywords.update(name.split())
        keywords.update(word for word in desc.split() if len(word) > 3)
    
    def score(u: str) -> int:
        path = urlparse(u).path.lower()
        return sum(1 for kw in keywords if kw in path)
    
    sorted_urls = sorted(urls, key=score, reverse=True)
    return [{"url": u, "relevance_score": 5} for u in sorted_urls[:limit]]


# =============================================================================
# Node: Scrape Pages
# =============================================================================

def scrape_pages_node(state: HTMLMinerState) -> dict:
    """
    Scrapes selected pages and stores their markdown content.
    """
    url = state["url"]
    selected_pages = state.get("selected_pages", [])
    engine = state.get("engine", "firecrawl")
    session_id = state.get("session_id", "")
    firecrawl_api_key = state.get("firecrawl_api_key")
    actual_engine = "firecrawl" if engine == "firecrawl" and firecrawl_api_key else "trafilatura"
    
    # For simple mode, selected_pages might be empty - scrape the base URL
    if not selected_pages:
        selected_pages = [{"url": url, "relevance_score": 10}]
    
    cached_by_url: dict[str, str] = {}
    urls_to_scrape: list[str] = []
    
    if state.get("status_callback"):
         state["status_callback"](f"Checking cache for {len(selected_pages)} pages...")

    for page in selected_pages:
        page_url = page["url"]
        cached_content = _get_cached_snapshot(
            page_url,
            actual_engine,
            session_id,
            state.get("status_callback"),
        )
        if cached_content is not None:
            cached_by_url[page_url] = cached_content
        else:
            urls_to_scrape.append(page_url)
    
    if state.get("status_callback"):
        if cached_by_url:
            state["status_callback"](
                f"Scraping {len(urls_to_scrape)} pages (using {len(cached_by_url)} cached)..."
            )
        else:
            state["status_callback"](f"Scraping {len(urls_to_scrape)} pages...")
    
    with StepTimingWrapper("scrape_pages", session_id, url, status_callback=state.get("status_callback")) as timer:
        scraped = []
        
        results_by_url: dict[str, str] = {}

        if actual_engine == "firecrawl":
            if urls_to_scrape:
                # Use Firecrawl batch scraping
                results = scrape_firecrawl_urls(
                    urls=urls_to_scrape,
                    api_key=firecrawl_api_key,
                    session_id=session_id,
                    status_callback=state.get("status_callback"),
                )
                for page_url, content in results:
                    results_by_url[page_url] = content
        else:
            # Use trafilatura for each URL
            for page in selected_pages:
                if page["url"] in cached_by_url:
                    continue
                if page["url"] not in urls_to_scrape:
                    continue
                try:
                    results = crawl_domain(
                        url=page["url"],
                        engine="trafilatura",
                        limit=1,
                        session_id=session_id,
                    )
                    if results:
                        results_by_url[page["url"]] = results[0][1]
                except Exception as e:
                    log_event(session_id, "graph", "WARNING", f"Failed to scrape {page['url']}: {e}")

        failed_urls = []
        for page in selected_pages:
            page_url = page["url"]
            content = cached_by_url.get(page_url) or results_by_url.get(page_url)
            if not content:
                failed_urls.append(page_url)
                log_event(session_id, "graph", "WARNING", f"No content returned for {page_url}")
                continue
            scraped.append(
                {
                    "url": page_url,
                    "content": content,
                    "relevance_score": page.get("relevance_score", 5),
                }
            )

        timer.set_details(
            {"scraped_count": len(scraped), "cached_count": len(cached_by_url), "failed_count": len(failed_urls)}
        )

    log_event(session_id, "graph", "INFO", f"Scraped {len(scraped)} pages")

    # Build warnings for CLI display
    scrape_warnings = []
    if failed_urls:
        warning_msg = f"Failed to retrieve content from {len(failed_urls)} page(s): {', '.join(failed_urls[:3])}{'...' if len(failed_urls) > 3 else ''}"
        scrape_warnings.append(warning_msg)
        console.print(f"[yellow]Warning:[/yellow] {warning_msg}")
    if not scraped:
        error_msg = "No page content was retrieved. The site may be blocking scrapers or returning empty responses."
        scrape_warnings.append(error_msg)
        console.print(f"[red]Error:[/red] {error_msg}")
        log_event(session_id, "graph", "ERROR", "Scraping returned no content for any selected pages", {"failed_urls": failed_urls})

    return {"scraped_pages": scraped, "scrape_warnings": scrape_warnings}


# =============================================================================
# Helper: Create Feature Examples for LangExtract
# =============================================================================

def _create_feature_examples(feature_name: str, feature_desc: str) -> list:
    """
    Creates ExampleData objects for langextract based on the feature type.
    LangExtract requires examples to guide the extraction.
    """
    # Generic examples that work for most extraction scenarios
    examples = {
        "Risk": [
            ExampleData(
                text="AI systems could pose significant risks if not developed carefully. Existential risks from advanced AI are a key concern.",
                extractions=[
                    Extraction(extraction_class="Risk", extraction_text="significant risks if not developed carefully"),
                    Extraction(extraction_class="Risk", extraction_text="Existential risks from advanced AI"),
                ],
            ),
        ],
        "Goal": [
            ExampleData(
                text="Our mission is to develop safe AI systems. We aim to achieve beneficial AI alignment for humanity.",
                extractions=[
                    Extraction(extraction_class="Goal", extraction_text="develop safe AI systems"),
                    Extraction(extraction_class="Goal", extraction_text="achieve beneficial AI alignment for humanity"),
                ],
            ),
        ],
        "Method": [
            ExampleData(
                text="We use interpretability research and constitutional AI training methods. Our approach includes red-teaming and safety evaluations.",
                extractions=[
                    Extraction(extraction_class="Method", extraction_text="interpretability research"),
                    Extraction(extraction_class="Method", extraction_text="constitutional AI training methods"),
                    Extraction(extraction_class="Method", extraction_text="red-teaming and safety evaluations"),
                ],
            ),
        ],
    }
    
    # Return feature-specific examples if available, otherwise create a generic one
    if feature_name in examples:
        return examples[feature_name]
    
    # Generic fallback example
    return [
        ExampleData(
            text=f"This document discusses {feature_desc.lower()}. The key point is that we focus on this important aspect.",
            extractions=[
                Extraction(extraction_class=feature_name, extraction_text=f"key point is that we focus on this important aspect"),
            ],
        ),
    ]




# =============================================================================
# Node: Extract Features (per page, coarse mode)
# =============================================================================

def extract_pages_node(state: HTMLMinerState) -> dict:
    """
    Runs LangExtract on each page separately (coarse mode).
    Each page is processed as a separate tool call.
    """
    scraped_pages = state.get("scraped_pages", [])
    features = state.get("features", [])
    api_key = state.get("api_key", "")
    model_tier = state.get("model_tier", "cheap")
    session_id = state.get("session_id", "")
    url = state.get("url", "")
    max_char_buffer = state.get("langextract_max_char_buffer", LANGEXTRACT_MAX_CHAR_BUFFER)
    if not isinstance(max_char_buffer, int) or max_char_buffer <= 0:
        max_char_buffer = LANGEXTRACT_MAX_CHAR_BUFFER
    fast_mode = state.get("langextract_fast", False)
    max_workers = state.get("langextract_max_workers", 1)
    if not isinstance(max_workers, int) or max_workers < 1:
        max_workers = 1
    
    if not scraped_pages or not features:
        return {"page_extractions": {}, "raw_counts": {}}
    
    model_id = MODEL_TIERS.get(model_tier, MODEL_TIERS["cheap"]).get("langextract_model_id", "gemini-2.0-flash")
    page_extractions = {}
    raw_counts = {f["name"]: 0 for f in features}
    
    # Print extraction plan with page sizes and time estimate
    valid_pages = [p for p in scraped_pages if p.get("content") and len(p.get("content", "").strip()) >= 100]
    if valid_pages and state.get("status_callback"):
        console.print(f"\n[bold cyan]LangExtract Plan:[/bold cyan]")
        total_chars = 0
        for idx, page in enumerate(valid_pages, 1):
            content = page.get("content", "")
            char_count = len(content)
            total_chars += char_count
            # Estimate: ~5s per feature if content < buffer, else ~5s per chunk
            chunks = max(1, char_count // max_char_buffer)
            console.print(f"  {idx}. {page['url'][:60]}... ({char_count:,} chars, {chunks} chunk{'s' if chunks > 1 else ''})")
        
        calls_per_page = 1 if fast_mode else len(features)
        total_api_calls = len(valid_pages) * calls_per_page
        # ~5s per API call average
        est_seconds = total_api_calls * 5
        est_minutes = est_seconds / 60
        calls_label = "1 call (fast mode)" if fast_mode else f"{len(features)} features"
        console.print(f"\n[dim]  Total: {len(valid_pages)} pages × {calls_label} = {total_api_calls} API calls[/dim]")
        console.print(f"[dim]  Estimated time: ~{est_minutes:.1f} minutes ({total_chars:,} total chars)[/dim]\n")
    
    with StepTimingWrapper("extract_pages", session_id, url, status_callback=state.get("status_callback")) as timer:
        for i, page in enumerate(scraped_pages):
            page_url = page["url"]
            content = page.get("content", "")
            
            if not content or len(content.strip()) < 100:
                continue
            
            page_extractions[page_url] = {}
            page_start_time = time.time()
            feature_timings = []
            
            if fast_mode:
                if state.get("status_callback"):
                    kb_size = len(content) / 1024
                    state["status_callback"](f"Extracting all features from page {i+1}/{len(scraped_pages)} ({kb_size:.1f}KB)...")

                feature_start_time = time.time()
                with StepTimingWrapper("extract_fast", session_id, page_url):
                    try:
                        feature_lines = "\n".join([
                            f"- {f['name']}: {f['description']}"
                            for f in features
                        ])
                        prompt_desc = (
                            "Extract the following features from the text. "
                            "Label each extraction with its feature name:\n"
                            f"{feature_lines}"
                        )
                        combined_examples = []
                        for feature in features:
                            combined_examples.extend(
                                _create_feature_examples(feature["name"], feature["description"])
                            )

                        annotated_doc = lx.extract(
                            content,
                            prompt_description=prompt_desc,
                            model_id=model_id,
                            api_key=api_key,
                            examples=combined_examples,
                            extraction_passes=LANGEXTRACT_EXTRACTION_PASSES,
                            max_char_buffer=max_char_buffer,
                            max_workers=max_workers,
                            batch_length=max_workers,
                            debug=False,
                            show_progress=False,
                            resolver_params={"enable_fuzzy_alignment": False},
                        )

                        feature_elapsed = time.time() - feature_start_time

                        by_feature = {f["name"]: [] for f in features}
                        for extraction in annotated_doc.extractions:
                            if not extraction.extraction_text:
                                continue
                            if len(extraction.extraction_text.split()) < 5:
                                continue
                            feature_name = extraction.extraction_class
                            if feature_name in by_feature:
                                by_feature[feature_name].append(extraction.extraction_text)

                        for feature_name, snippets in by_feature.items():
                            snippets = list(dict.fromkeys(snippets))
                            by_feature[feature_name] = snippets
                            page_extractions[page_url][feature_name] = snippets
                            raw_counts[feature_name] += len(snippets)
                            feature_timings.append((feature_name, None, len(snippets)))

                        if state.get("status_callback"):
                            state["status_callback"](f"  ✓ Fast extraction in {feature_elapsed:.2f}s")

                        total_input_chars = len(content) + len(prompt_desc)
                        total_output_chars = sum(len(s) for snippets in by_feature.values() for s in snippets)
                        est_prompt_tokens = total_input_chars // 4
                        est_completion_tokens = total_output_chars // 4

                        current_tokens = timer.details.get("tokens", {
                            "prompt_tokens": 0, "completion_tokens": 0, "total_calls": 0
                        })
                        current_tokens["prompt_tokens"] += est_prompt_tokens
                        current_tokens["completion_tokens"] += est_completion_tokens
                        current_tokens["total_calls"] += 1
                        timer.set_details({"tokens": current_tokens})
                    except Exception as e:
                        feature_elapsed = time.time() - feature_start_time
                        if state.get("status_callback"):
                            state["status_callback"](f"  ✗ Fast extraction failed in {feature_elapsed:.2f}s")
                        log_event(
                            session_id, "graph", "WARNING",
                            f"Fast extract failed on {page_url}: {e}"
                        )
                        for feature in features:
                            feature_name = feature["name"]
                            feature_timings.append((feature_name, None, 0))
                            page_extractions[page_url][feature_name] = []
            else:
                # Process each feature separately (sequential for now)
                for feature in features:
                    feature_name = feature["name"]
                    feature_desc = feature["description"]
                    
                    if state.get("status_callback"):
                        kb_size = len(content) / 1024
                        state["status_callback"](f"Extracting '{feature_name}' from page {i+1}/{len(scraped_pages)} ({kb_size:.1f}KB)...")
                    
                    feature_start_time = time.time()
                    with StepTimingWrapper(f"extract_{feature_name}", session_id, page_url):
                        try:
                            # Create examples for this feature
                            feature_examples = _create_feature_examples(feature_name, feature_desc)
                            
                            # Let langextract handle its own chunking internally
                            # Disable fuzzy_alignment which has O(n³) complexity and causes hangs
                            annotated_doc = lx.extract(
                                content,  # Pass full content, langextract handles chunking
                                prompt_description=feature_desc,
                                model_id=model_id,
                                api_key=api_key,
                                examples=feature_examples,
                                extraction_passes=LANGEXTRACT_EXTRACTION_PASSES,
                                max_char_buffer=max_char_buffer,
                                max_workers=max_workers,
                                batch_length=max_workers,
                                debug=False,
                                show_progress=False,  # Disable progress bar output
                                resolver_params={"enable_fuzzy_alignment": False},  # Avoid O(n³) alignment
                            )
                            
                            feature_elapsed = time.time() - feature_start_time
                            
                            # Extract snippets from AnnotatedDocument
                            snippets = [
                                e.extraction_text for e in annotated_doc.extractions
                                if e.extraction_text and len(e.extraction_text.split()) >= 5
                            ]
                            # Deduplicate
                            snippets = list(dict.fromkeys(snippets))
                            
                            page_extractions[page_url][feature_name] = snippets
                            raw_counts[feature_name] += len(snippets)
                            
                            # Show feature-level timing in status callback
                            if state.get("status_callback"):
                                state["status_callback"](f"  ✓ '{feature_name}' extracted in {feature_elapsed:.2f}s ({len(snippets)} snippets)")
                            feature_timings.append((feature_name, feature_elapsed, len(snippets)))
                            
                            # Estimate usage (approx 4 chars/token)
                            total_input_chars = len(content) + len(feature_desc)
                            total_output_chars = sum(len(s) for s in snippets)
                            est_prompt_tokens = total_input_chars // 4
                            est_completion_tokens = total_output_chars // 4
                            
                            # Aggregate for timer details
                            current_tokens = timer.details.get("tokens", {
                                "prompt_tokens": 0, "completion_tokens": 0, "total_calls": 0
                            })
                            current_tokens["prompt_tokens"] += est_prompt_tokens
                            current_tokens["completion_tokens"] += est_completion_tokens
                            current_tokens["total_calls"] += 1
                            timer.set_details({"tokens": current_tokens})

                        except Exception as e:
                            feature_elapsed = time.time() - feature_start_time
                            feature_timings.append((feature_name, feature_elapsed, 0))
                            # Show feature-level timing for failures too
                            if state.get("status_callback"):
                                state["status_callback"](f"  ✗ '{feature_name}' failed in {feature_elapsed:.2f}s")
                            log_event(
                                session_id, "graph", "WARNING",
                                f"Extract failed for {feature_name} on {page_url}: {e}"
                            )
                            page_extractions[page_url][feature_name] = []
            
            # Print page extraction summary
            page_elapsed = time.time() - page_start_time
            short_url = page_url[:50] + "..." if len(page_url) > 50 else page_url
            if fast_mode:
                feature_summary = " | ".join([f"{name}: {count}" for name, _, count in feature_timings])
            else:
                feature_summary = " | ".join([f"{name}: {elapsed:.1f}s ({count})" for name, elapsed, count in feature_timings])
            console.print(f"  [green]✓[/green] Page {i+1}: {short_url} [dim]({page_elapsed:.1f}s total)[/dim]")
            console.print(f"    [dim]{feature_summary}[/dim]")
        
        timer.set_details({
            "page_count": len(page_extractions),
            "raw_counts": raw_counts,
        })
    
    return {"page_extractions": page_extractions, "raw_counts": raw_counts}


# =============================================================================
# Node: Synthesize Results
# =============================================================================

# =============================================================================
# Router: Check Extraction
# =============================================================================

def route_extraction(state: HTMLMinerState) -> str:
    """
    Conditional edge function: determines whether to run extraction or skip to synthesis.
    """
    if state.get("use_langextract", False):
        return "extract"
    return "synthesize"


# =============================================================================
# Node: Synthesize Results
# =============================================================================

def synthesize_node(state: HTMLMinerState) -> dict:
    """
    Synthesizes extractions from all pages into final results per feature.
    If use_langextract is False, uses full page content instead of snippets.
    """
    page_extractions = state.get("page_extractions", {})
    scraped_pages = state.get("scraped_pages", [])
    features = state.get("features", [])
    api_key = state.get("api_key", "")
    model_tier = state.get("model_tier", "cheap")
    use_langextract = state.get("use_langextract", False)
    synthesis_top = state.get("synthesis_top", 50)
    session_id = state.get("session_id", "")
    url = state.get("url", "")
    raw_counts = state.get("raw_counts", {})
    
    if not features:
        log_event(session_id, "graph", "WARNING", "No features configured - skipping synthesis")
        if state.get("status_callback"):
            state["status_callback"]("No features configured - skipping synthesis")
        return {"results": {"URL": url, "Counts": {}}}

    # If using langextract, we need extractions. If not, we need scraped pages.
    if use_langextract and not page_extractions:
        log_event(session_id, "graph", "WARNING", "No extractions available for synthesis")
        if state.get("status_callback"):
            state["status_callback"]("No extractions available - cannot synthesize results")
        return {"results": {"URL": url, "Counts": {}}}
    if not use_langextract and not scraped_pages:
        log_event(session_id, "graph", "WARNING", "No scraped content available for synthesis")
        if state.get("status_callback"):
            state["status_callback"]("No scraped content available - cannot synthesize results")
        return {"results": {"URL": url, "Counts": {}}}

    model_name = MODEL_TIERS.get(model_tier, MODEL_TIERS["cheap"])["langchain_model"]
    results = {"URL": url, "Counts": raw_counts}
    
    if state.get("status_callback"):
        mode_msg = "snippets" if use_langextract else "full content"
        state["status_callback"](f"Synthesizing results from {mode_msg}...")
    
    with StepTimingWrapper("synthesize", session_id, url, status_callback=state.get("status_callback")) as timer:
        callback = TimingTokenCallback(session_id, "synthesize_llm")
        
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
        )
        structured_llm = llm.with_structured_output(FeatureSynthesis)
        
        # Prepare content for synthesis
        if not use_langextract:
            # Join content from all pages, maybe with some delimiter
            # Gemini has large context, so we can concatenate.
            combined_content = ""
            for p in scraped_pages:
                 combined_content += f"\n\n--- Page: {p['url']} ---\n{p.get('content', '')}"
        
        for feature in features:
            feature_name = feature["name"]
            synthesis_topic = feature.get("synthesis_topic", feature_name)
            length_guidance = feature.get("length", "")
            if not isinstance(length_guidance, str):
                length_guidance = ""
            length_guidance = length_guidance.strip()
            output_format = feature.get("output_format", "")
            if not isinstance(output_format, str):
                output_format = ""
            output_format = output_format.strip().lower()
            output_categories_raw = feature.get("output_categories", [])
            if isinstance(output_categories_raw, str):
                output_categories = [c.strip() for c in output_categories_raw.split(",") if c.strip()]
            elif isinstance(output_categories_raw, (list, tuple)):
                output_categories = [str(c).strip() for c in output_categories_raw if str(c).strip()]
            else:
                output_categories = []
            
            prompt_content = ""
            
            if use_langextract:
                # Collect all snippets for this feature across pages
                all_snippets = []
                for page_url, extractions in page_extractions.items():
                    snippets = extractions.get(feature_name, [])
                    for s in snippets:
                        all_snippets.append(f"[From {page_url}]: {s}")
                
                if not all_snippets:
                    results[feature_name] = "Not mentioned"
                    continue
                
                # Take top N longest snippets
                all_snippets.sort(key=len, reverse=True)
                top_snippets = all_snippets[:synthesis_top]
                prompt_content = "Extractions:\n" + "\n".join(top_snippets)
            else:
                # Use full content
                prompt_content = f"Source Content:\n{combined_content}"

            prompt_lines = [
                f'Synthesize the following information about "{synthesis_topic}" into a coherent summary.',
                "Be concise and factual.",
            ]
            if length_guidance:
                prompt_lines.append(f"Length guidance: {length_guidance}")
            if output_format:
                if output_format in {"number", "numeric", "integer", "float"}:
                    prompt_lines.append("Output format: number only (no units or extra text).")
                elif is_list_output_format(output_format):
                    prompt_lines.append(list_output_instruction())
                elif output_format in {"free_text", "text"}:
                    prompt_lines.append("Output format: free text.")
                elif output_format in {"single_choice", "single_choice_category", "single"}:
                    if output_categories:
                        prompt_lines.append(
                            f"Output format: single choice from categories: {', '.join(output_categories)}."
                        )
                    else:
                        prompt_lines.append("Output format: single choice.")
                elif output_format in {"multi_choice", "multiple_choice", "multi"}:
                    if output_categories:
                        prompt_lines.append(
                            "Output format: multiple choices (comma-separated) from categories: "
                            f"{', '.join(output_categories)}."
                        )
                    else:
                        prompt_lines.append("Output format: multiple choices (comma-separated).")
                else:
                    prompt_lines.append(f"Output format: {output_format}.")
            prompt_lines.append("")
            prompt_lines.append(prompt_content)
            prompt = "\n".join(prompt_lines)

            try:
                result = structured_llm.invoke(
                    [HumanMessage(content=prompt)],
                    config={"callbacks": [callback]},
                )
                summary = result.summary
                if is_list_output_format(output_format):
                    summary = normalize_bullet_list(summary)
                results[feature_name] = summary
            except Exception as e:
                log_event(session_id, "graph", "ERROR", f"Synthesis failed for {feature_name}: {e}")
                results[feature_name] = f"Error: {str(e)}"
        
        timer.set_details({
            "feature_count": len(features),
            "tokens": callback.get_totals(),
            "mode": "langextract" if use_langextract else "full_content"
        })
    
    return {"results": results}


# =============================================================================
# Node: Simple Crawl (for non-smart mode)
# =============================================================================

def simple_crawl_node(state: HTMLMinerState) -> dict:
    """
    Simple crawling for non-smart mode: just scrape the base URL and nearby pages.
    """
    url = state["url"]
    engine = state.get("engine", "trafilatura")
    limit = state.get("limit", 1)
    session_id = state.get("session_id", "")
    firecrawl_api_key = state.get("firecrawl_api_key")
    actual_engine = "firecrawl" if engine == "firecrawl" and firecrawl_api_key else "trafilatura"
    
    if state.get("status_callback"):
        state["status_callback"](f"Crawling {url}...")
    
    with StepTimingWrapper("simple_crawl", session_id, url, status_callback=state.get("status_callback")) as timer:
        if limit == 1:
            cached_content = _get_cached_snapshot(
                url,
                actual_engine,
                session_id,
                state.get("status_callback"),
            )
            if cached_content is not None:
                scraped = [{"url": url, "content": cached_content, "relevance_score": 10}]
                timer.set_details({"scraped_count": len(scraped), "cached_count": 1})
                return {"scraped_pages": scraped}
        
        results = crawl_domain(
            url=url,
            engine=actual_engine,
            limit=limit,
            session_id=session_id,
            status_callback=state.get("status_callback"),
        )
        
        scraped = [
            {"url": page_url, "content": content, "relevance_score": 10}
            for page_url, content in results
        ]
        
        timer.set_details({"scraped_count": len(scraped)})
    
    return {"scraped_pages": scraped}
