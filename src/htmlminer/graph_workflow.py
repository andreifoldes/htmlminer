"""
HTMLMiner Graph Workflow

Builds and compiles the LangGraph workflow for HTMLMiner.
"""

from langgraph.graph import StateGraph, START, END

from .graph_state import HTMLMinerState
from .graph_nodes import (
    check_engine_node,
    route_by_engine,
    fetch_sitemap_node,
    filter_urls_node,
    select_pages_node,
    scrape_pages_node,
    extract_pages_node,
    synthesize_node,
    simple_crawl_node,
    route_extraction,
)


def build_htmlminer_graph():
    """
    Builds and compiles the HTMLMiner workflow graph.
    
    The workflow implements the decision tree:
    
    START -> check_engine -> [smart: fetch_sitemap] or [simple: simple_crawl]
    
    Smart path:
        fetch_sitemap -> filter_urls -> select_pages -> scrape_pages -> [route_extraction]
    
    Simple path:
        simple_crawl -> [route_extraction]

    [route_extraction]:
        - extract: extract_pages -> synthesize -> END
        - synthesize: synthesize -> END
    
    Returns:
        CompiledGraph: The compiled LangGraph workflow
    """
    builder = StateGraph(HTMLMinerState)
    
    # Add all nodes
    builder.add_node("check_engine", check_engine_node)
    builder.add_node("fetch_sitemap", fetch_sitemap_node)
    builder.add_node("filter_urls", filter_urls_node)
    builder.add_node("select_pages", select_pages_node)
    builder.add_node("scrape_pages", scrape_pages_node)
    builder.add_node("simple_crawl", simple_crawl_node)
    builder.add_node("extract_pages", extract_pages_node)
    builder.add_node("synthesize", synthesize_node)
    
    # Add edges
    builder.add_edge(START, "check_engine")
    
    # Conditional routing based on engine/smart mode
    builder.add_conditional_edges(
        "check_engine",
        route_by_engine,
        {
            "smart": "fetch_sitemap",
            "simple": "simple_crawl",
        }
    )
    
    # Smart crawling path
    builder.add_edge("fetch_sitemap", "filter_urls")
    builder.add_edge("filter_urls", "select_pages")
    builder.add_edge("select_pages", "scrape_pages")
    
    # Smart path: Route to extraction or direct synthesis
    builder.add_conditional_edges(
        "scrape_pages",
        route_extraction,
        {
            "extract": "extract_pages",
            "synthesize": "synthesize",
        }
    )
    
    # Simple crawling path: Route to extraction or direct synthesis
    builder.add_conditional_edges(
        "simple_crawl",
        route_extraction,
        {
            "extract": "extract_pages",
            "synthesize": "synthesize",
        }
    )
    
    # Extraction path
    builder.add_edge("extract_pages", "synthesize")
    
    # Final step
    builder.add_edge("synthesize", END)
    
    return builder.compile()


def run_htmlminer_workflow(
    url: str,
    features: list[dict],
    api_key: str,
    firecrawl_api_key: str = None,
    engine: str = "firecrawl",
    smart_mode: bool = True,
    limit: int = 10,
    model_tier: str = "cheap",
    max_paragraphs: int = 3,
    session_id: str = None,
    status_callback: callable = None,
    use_langextract: bool = False,
    langextract_max_char_buffer: int = None,
    synthesis_top: int = 50,
) -> dict:
    """
    Convenience function to run the HTMLMiner workflow.
    
    Args:
        url: Base URL to analyze
        features: List of feature configs from config.json
        api_key: Gemini API key
        firecrawl_api_key: Firecrawl API key (optional)
        engine: 'firecrawl' or 'trafilatura'
        smart_mode: Whether to use smart crawling
        limit: Max pages to select
        model_tier: 'cheap' or 'expensive'
        max_paragraphs: Max paragraphs per feature in synthesis
        session_id: Session ID for logging
        status_callback: Callback for status updates
        use_langextract: Whether to use LangExtract for intermediate extraction
        langextract_max_char_buffer: Max chars per chunk for LangExtract (optional)
        synthesis_top: Max snippets per feature for synthesis
    
    Returns:
        dict: The final workflow state containing results
    """
    graph = build_htmlminer_graph()
    
    initial_state = {
        "url": url,
        "features": features,
        "api_key": api_key,
        "firecrawl_api_key": firecrawl_api_key,
        "engine": engine,
        "smart_mode": smart_mode,
        "limit": limit,
        "model_tier": model_tier,
        "max_paragraphs": max_paragraphs,
        "session_id": session_id or "",
        "status_callback": status_callback,
        "use_langextract": use_langextract,
        "synthesis_top": synthesis_top,
    }
    if langextract_max_char_buffer is not None:
        initial_state["langextract_max_char_buffer"] = langextract_max_char_buffer
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    return final_state


def get_workflow_diagram() -> str:
    """
    Returns a Mermaid diagram of the workflow for documentation.
    """
    graph = build_htmlminer_graph()
    return graph.get_graph().draw_mermaid()
