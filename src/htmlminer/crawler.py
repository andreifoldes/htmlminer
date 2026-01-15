import trafilatura
import os
from firecrawl import FirecrawlApp

from .database import log_event, save_snapshot

def crawl_and_snapshot(url: str, engine: str = "trafilatura", session_id: str = None) -> str:
    """
    Downloads and converts a webpage to Markdown using the specified engine.
    """
    if session_id:
        log_event(session_id, "crawler", "INFO", f"Starting snapshot for {url} with engine {engine}")
    
    content = None
    if engine == "firecrawl":
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise Exception("FIRECRAWL_API_KEY environment variable not set. Please set it in .env or environment.")
        
        try:
            app = FirecrawlApp(api_key=api_key)
            result = app.scrape(url, formats=['markdown'])
            if not result or not hasattr(result, 'markdown') or not result.markdown:
                raise Exception(f"Firecrawl failed to return markdown for {url}")
            content = result.markdown
        except Exception as e:
             raise Exception(f"Firecrawl error: {e}")

    elif engine == "trafilatura":
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise Exception(f"Failed to fetch {url}")
        
        result = trafilatura.extract(downloaded, include_links=True, include_images=False, include_tables=True, output_format='markdown')
        if not result:
            raise Exception(f"Failed to extract content from {url}")
            
        content = result
    
    else:
        raise ValueError(f"Unknown engine: {engine}")

    if content:
        save_snapshot(url, engine, content)
        
    return content
