"""
Firecrawl Agent SDK integration for HTMLMiner.

This module provides a workflow using Firecrawl's Agent API with Spark 1 models
for autonomous web data extraction.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field, create_model
from firecrawl import FirecrawlApp
from rich.console import Console

from .database import log_event

console = Console()

# Available Spark models
SPARK_MODELS = {
    "mini": "spark-1-mini",
    "pro": "spark-1-pro",
}


class FirecrawlAgentExtractor:
    """
    Extracts data from URLs using Firecrawl's Agent API.
    
    The Agent API autonomously searches, navigates, and gathers data
    using Spark 1 models (mini or pro).
    """

    def __init__(
        self,
        api_key: str,
        spark_model: str = "mini",
        session_id: Optional[str] = None,
        step_timeout_s: Optional[int] = None,
    ):
        if spark_model not in SPARK_MODELS:
            raise ValueError(
                f"Invalid spark_model '{spark_model}'. Choose from: {', '.join(SPARK_MODELS.keys())}"
            )
        
        self.api_key = api_key
        self.spark_model = spark_model
        self.model_id = SPARK_MODELS[spark_model]
        self.session_id = session_id
        self.step_timeout_s = step_timeout_s if step_timeout_s and step_timeout_s > 0 else None
        
        self.app = FirecrawlApp(api_key=api_key)

    def build_schema_from_config(self, features: list) -> type[BaseModel]:
        """
        Dynamically build a Pydantic model from config.json features.
        
        Args:
            features: List of feature dicts with 'name' and 'description' keys.
            
        Returns:
            A dynamically created Pydantic BaseModel class.
        """
        field_definitions = {}
        
        for feature in features:
            name = feature["name"]
            description = feature.get("description", "")
            # Each feature becomes an optional string field
            field_definitions[name] = (
                Optional[str],
                Field(None, description=description)
            )
        
        # Create the model dynamically
        DynamicSchema = create_model("ExtractionSchema", **field_definitions)
        return DynamicSchema

    def build_prompt_from_config(self, url: str, features: list) -> str:
        """
        Build a prompt for the agent from the config features.
        
        Args:
            url: The target URL.
            features: List of feature dicts.
            
        Returns:
            A formatted prompt string.
        """
        feature_descriptions = []
        for f in features:
            name = f["name"]
            desc = f.get("description", name)
            feature_descriptions.append(f"- {name}: {desc}")
        
        features_text = "\n".join(feature_descriptions)
        
        prompt = f"""Analyze the website at {url} and extract the following information:

{features_text}

Be thorough and look for this information across the entire website, including about pages, research sections, and any relevant subpages."""
        
        return prompt

    def run(
        self,
        url: str,
        features: list,
        status_callback=None,
    ) -> dict:
        """
        Run the Firecrawl Agent to extract data from a URL.
        
        Args:
            url: The URL to analyze.
            features: List of feature dicts from config.json.
            status_callback: Optional callback for status updates.
            
        Returns:
            A dict with extracted data including URL and feature values.
        """
        if status_callback:
            status_callback(f"Building schema for {len(features)} features...")
        
        # Build dynamic schema from config
        schema = self.build_schema_from_config(features)
        
        # Build prompt
        prompt = self.build_prompt_from_config(url, features)
        
        if self.session_id:
            log_event(
                self.session_id,
                "firecrawl_agent",
                "INFO",
                f"Starting agent extraction for {url}",
                {"model": self.model_id, "features": [f["name"] for f in features]},
            )
        
        if status_callback:
            status_callback(f"Running Firecrawl Agent ({self.model_id})...")
        
        console.print(f"[dim]Using Firecrawl Agent with model: {self.model_id}[/dim]")
        
        try:
            # Call the Firecrawl Agent API
            result = self.app.agent(
                prompt=prompt,
                schema=schema,
                model=self.model_id,
                urls=[url],
                timeout=self.step_timeout_s,
            )
            
            if status_callback:
                status_callback("Processing agent response...")
            
            # Extract data from result
            data = result.data if hasattr(result, 'data') else result
            
            # Capture visited pages from the agent response
            visited_pages = []
            if hasattr(result, 'sources'):
                visited_pages = result.sources
            elif hasattr(result, 'urls'):
                visited_pages = result.urls
            elif hasattr(result, 'urlTrace'):
                visited_pages = result.urlTrace
            elif hasattr(result, 'visited_urls'):
                visited_pages = result.visited_urls
            
            # Convert to dict format matching expected output
            result_data = {
                "URL": url,
                "Agent_Visited_Pages": visited_pages if visited_pages else [url],
            }
            
            if isinstance(data, dict):
                for feature in features:
                    name = feature["name"]
                    result_data[name] = data.get(name, "Not found.")
                    # Store as structured object with text and source
                    raw_value = data.get(name, "")
                    result_data[f"{name}_Raw"] = [{"text": raw_value, "source": url}] if raw_value else []
                    result_data[f"{name}_Count"] = 1 if data.get(name) else 0
            elif hasattr(data, '__dict__'):
                # Pydantic model instance
                for feature in features:
                    name = feature["name"]
                    value = getattr(data, name, None)
                    result_data[name] = value if value else "Not found."
                    # Store as structured object with text and source
                    result_data[f"{name}_Raw"] = [{"text": value, "source": url}] if value else []
                    result_data[f"{name}_Count"] = 1 if value else 0
            else:
                # Fallback - just store raw result
                result_data["Raw_Response"] = str(data)
            
            # Log all available attributes for debugging
            if self.session_id:
                available_attrs = [attr for attr in dir(result) if not attr.startswith('_')]
                log_event(
                    self.session_id,
                    "firecrawl_agent",
                    "DEBUG",
                    f"Agent result attributes: {available_attrs}",
                    {"visited_pages": visited_pages},
                )
            
            if self.session_id:
                log_event(
                    self.session_id,
                    "firecrawl_agent",
                    "INFO",
                    f"Agent extraction complete for {url}",
                    {"result_keys": list(result_data.keys()), "visited_pages_count": len(visited_pages)},
                )
            
            return result_data
            
        except Exception as e:
            console.print(f"[red]Firecrawl Agent error: {e}[/red]")
            if self.session_id:
                log_event(
                    self.session_id,
                    "firecrawl_agent",
                    "ERROR",
                    f"Agent extraction failed for {url}",
                    {"error": str(e)},
                )
            raise
