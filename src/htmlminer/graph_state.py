"""
HTMLMiner Graph State Schema

Defines the state that flows through the LangGraph workflow.
"""

from typing import TypedDict, Optional
from typing_extensions import Annotated
import operator


class PageInfo(TypedDict):
    """Information about a scraped page."""
    url: str
    content: str
    relevance_score: Optional[int]


class StepTiming(TypedDict):
    """Timing information for a workflow step."""
    step_name: str
    duration_seconds: float
    prompt_tokens: int
    completion_tokens: int
    details: Optional[dict]


class HTMLMinerState(TypedDict, total=False):
    """
    State schema for the HTMLMiner workflow.
    
    All fields are optional (total=False) to allow incremental updates.
    """
    # === Input Configuration ===
    url: str  # Base URL to analyze
    engine: str  # 'firecrawl' or 'trafilatura'
    smart_mode: bool  # Whether to use smart crawling
    limit: int  # Max pages to select
    features: list[dict]  # Feature config from config.json
    session_id: str  # Session ID for logging
    api_key: str  # Gemini API key
    firecrawl_api_key: Optional[str]  # Firecrawl API key
    model_tier: str  # 'cheap' or 'expensive'
    model_tier: str  # 'cheap' or 'expensive'
    max_paragraphs: int  # Max paragraphs per feature in synthesis
    use_langextract: bool  # Whether to use langextract for intermediate extraction
    synthesis_top: int  # Max snippets per feature for synthesis

    
    # === Crawling Phase ===
    sitemap_urls: list[str]  # URLs from sitemap
    filtered_urls: list[str]  # URLs after domain/extension filtering
    
    # === Page Selection Phase ===
    selected_pages: list[dict]  # [{url, relevance_score}]
    relevance_scores: dict[str, dict[str, int]]  # {url: {feature: score}}
    
    # === Scraping Phase ===
    scraped_pages: list[PageInfo]  # Scraped page content
    
    # === Extraction Phase (coarse mode - per page) ===
    page_extractions: dict[str, dict[str, list[str]]]  # {url: {feature: [snippets]}}
    
    # === Synthesis Phase ===
    results: dict  # Final synthesized results {feature: summary, ...}
    raw_counts: dict[str, int]  # {feature: count} of raw extractions
    
    # === Tracking ===
    step_timings: Annotated[list[StepTiming], operator.add]  # Accumulates timing data
    total_tokens: dict[str, int]  # {prompt, completion, total}
    
    # === Status ===
    status_callback: Optional[callable]  # For CLI status updates
    error: Optional[str]  # Error message if workflow failed


# Model tier configuration (moved from agent.py)
MODEL_TIERS = {
    "cheap": {
        "model_id": "gemini-2.5-flash",
        "langchain_model": "gemini-2.5-flash",
    },
    "expensive": {
        "model_id": "gemini-2.5-pro",
        "langchain_model": "gemini-2.5-pro",
    },
}
