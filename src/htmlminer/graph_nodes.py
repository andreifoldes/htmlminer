"""
HTMLMiner Graph Nodes

Individual node implementations for the LangGraph workflow.
Each node takes the workflow state and returns updates to it.
"""

import time
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

from .graph_state import HTMLMinerState, MODEL_TIERS
from .graph_callbacks import StepTimingWrapper, TimingTokenCallback
from .crawler import (
    fetch_sitemap_urls,
    fetch_firecrawl_map_urls,
    scrape_firecrawl_urls,
    crawl_domain,
)
from .database import log_event, save_page_relevance

console = Console()

LANGEXTRACT_EXTRACTION_PASSES = 1
LANGEXTRACT_MAX_CHAR_BUFFER = 50000


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
    
    # For simple mode, selected_pages might be empty - scrape the base URL
    if not selected_pages:
        selected_pages = [{"url": url, "relevance_score": 10}]
    
    urls_to_scrape = [p["url"] for p in selected_pages]
    
    if state.get("status_callback"):
        state["status_callback"](f"Scraping {len(urls_to_scrape)} pages...")
    
    with StepTimingWrapper("scrape_pages", session_id, url, status_callback=state.get("status_callback")) as timer:
        scraped = []
        
        if engine == "firecrawl" and firecrawl_api_key:
            # Use Firecrawl batch scraping
            results = scrape_firecrawl_urls(
                urls=urls_to_scrape,
                api_key=firecrawl_api_key,
                session_id=session_id,
                status_callback=state.get("status_callback"),
            )
            for page_url, content in results:
                score = next(
                    (p["relevance_score"] for p in selected_pages if p["url"] == page_url),
                    5
                )
                scraped.append({
                    "url": page_url,
                    "content": content,
                    "relevance_score": score,
                })
        else:
            # Use trafilatura for each URL
            for page in selected_pages:
                try:
                    results = crawl_domain(
                        url=page["url"],
                        engine="trafilatura",
                        limit=1,
                        session_id=session_id,
                    )
                    if results:
                        scraped.append({
                            "url": page["url"],
                            "content": results[0][1],
                            "relevance_score": page.get("relevance_score", 5),
                        })
                except Exception as e:
                    log_event(session_id, "graph", "WARNING", f"Failed to scrape {page['url']}: {e}")
        
        timer.set_details({"scraped_count": len(scraped)})
    
    log_event(session_id, "graph", "INFO", f"Scraped {len(scraped)} pages")
    
    return {"scraped_pages": scraped}


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
    
    if not scraped_pages or not features:
        return {"page_extractions": {}, "raw_counts": {}}
    
    model_id = MODEL_TIERS.get(model_tier, MODEL_TIERS["cheap"])["model_id"]
    page_extractions = {}
    raw_counts = {f["name"]: 0 for f in features}
    
    with StepTimingWrapper("extract_pages", session_id, url, status_callback=state.get("status_callback")) as timer:
        for i, page in enumerate(scraped_pages):
            page_url = page["url"]
            content = page.get("content", "")
            
            if not content or len(content.strip()) < 100:
                continue
            
            if state.get("status_callback"):
                state["status_callback"](f"Extracting features from page {i+1}/{len(scraped_pages)}: {page_url[:50]}...")
            
            page_extractions[page_url] = {}
            
            # Process each feature separately (sequential for now)
            for feature in features:
                feature_name = feature["name"]
                feature_desc = feature["description"]
                
                with StepTimingWrapper(f"extract_{feature_name}", session_id, page_url):
                    try:
                        # Create examples for this feature
                        feature_examples = _create_feature_examples(feature_name, feature_desc)
                        
                        # Use LangExtract with correct API
                        annotated_doc = lx.extract(
                            content,  # text_or_documents (positional)
                            prompt_description=feature_desc,
                            model_id=model_id,
                            api_key=api_key,
                            examples=feature_examples,
                            extraction_passes=LANGEXTRACT_EXTRACTION_PASSES,
                            max_char_buffer=LANGEXTRACT_MAX_CHAR_BUFFER,
                        )
                        
                        # Extract snippets from AnnotatedDocument
                        snippets = [
                            e.extraction_text for e in annotated_doc.extractions
                            if e.extraction_text and len(e.extraction_text.split()) >= 5
                        ]
                        
                        # Deduplicate
                        snippets = list(dict.fromkeys(snippets))
                        
                        page_extractions[page_url][feature_name] = snippets
                        raw_counts[feature_name] += len(snippets)
                        
                        # Estimate usage (approx 4 chars/token)
                        # Input: Content + Prompt Description
                        input_chars = len(content) + len(feature_desc)
                        # Output: Snippets text
                        output_chars = sum(len(s) for s in snippets)
                        
                        est_prompt_tokens = input_chars // 4
                        est_completion_tokens = output_chars // 4
                        
                        # Aggregate for timer details
                        current_tokens = timer.details.get("tokens", {
                            "prompt_tokens": 0, "completion_tokens": 0, "total_calls": 0
                        })
                        current_tokens["prompt_tokens"] += est_prompt_tokens
                        current_tokens["completion_tokens"] += est_completion_tokens
                        current_tokens["total_calls"] += 1
                        timer.set_details({"tokens": current_tokens})

                    except Exception as e:
                        log_event(
                            session_id, "graph", "WARNING",
                            f"Extract failed for {feature_name} on {page_url}: {e}"
                        )
                        page_extractions[page_url][feature_name] = []
        
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
    max_paragraphs = state.get("max_paragraphs", 3)
    use_langextract = state.get("use_langextract", False)
    synthesis_top = state.get("synthesis_top", 50)
    session_id = state.get("session_id", "")
    url = state.get("url", "")
    raw_counts = state.get("raw_counts", {})
    
    if not features:
        return {"results": {"URL": url, "Counts": {}}}
    
    # If using langextract, we need extractions. If not, we need scraped pages.
    if use_langextract and not page_extractions:
        return {"results": {"URL": url, "Counts": {}}}
    if not use_langextract and not scraped_pages:
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

            prompt = f"""Synthesize the following information about "{synthesis_topic}" into a coherent summary.
Write at most {max_paragraphs} paragraphs. Be concise and factual.

{prompt_content}"""

            try:
                result = structured_llm.invoke(
                    [HumanMessage(content=prompt)],
                    config={"callbacks": [callback]},
                )
                results[feature_name] = result.summary
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
    
    if state.get("status_callback"):
        state["status_callback"](f"Crawling {url}...")
    
    with StepTimingWrapper("simple_crawl", session_id, url, status_callback=state.get("status_callback")) as timer:
        if engine == "firecrawl" and firecrawl_api_key:
            actual_engine = "firecrawl"
        else:
            actual_engine = "trafilatura"
        
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
