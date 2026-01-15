import os
import sys
import langextract as lx
import textwrap
import dspy
from typing import Optional
from rich.console import Console

# --- Monkey-patching for Token Usage ---
from langextract.providers.gemini import GeminiLanguageModel
from langextract.core import types as core_types
import threading

class TokenTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def update(self, prompt_tokens=0, completion_tokens=0):
        with self._lock:
            self.usage["prompt_tokens"] += prompt_tokens
            self.usage["completion_tokens"] += completion_tokens
            self.usage["total_tokens"] += (prompt_tokens + completion_tokens)
    
    def reset(self):
         with self._lock:
            self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def get_stats(self):
        with self._lock:
            return self.usage.copy()

# Global tracker instance
global_token_tracker = TokenTracker()

# Store original method
_original_process_single_prompt = GeminiLanguageModel._process_single_prompt

def _patched_process_single_prompt(self, prompt: str, config: dict) -> core_types.ScoredOutput:
    # We need to capture the response object to get usage metadata
    # The original method calls self._client.models.generate_content requesting 'usage_metadata' is implicit usually
    
    try:
        # Re-implementing parts of _process_single_prompt to access the response object directly
        # or we wrap the client call?
        # Wrapping the client call is cleaner but harder as client is initialized in __init__
        # Let's try to replicate the critical part or see if we can just wrap the return.
        # But _process_single_prompt returns ScoredOutput(text), swallowing the response object.
        # So we MUST re-implement or wrap carefully.
        
        # Let's look at how we can minimally duplicate logic.
        # The key line in original is:
        # response = self._client.models.generate_content(...)
        
        # We need to prepare config as original does
        for key, value in self._extra_kwargs.items():
             if key not in config and value is not None:
                config[key] = value

        if self.gemini_schema:
            # self._validate_schema_config() # internal method might be missing in some versions
            config.setdefault('response_mime_type', 'application/json')
            config.setdefault('response_schema', self.gemini_schema.schema_dict)

        # Call API
        response = self._client.models.generate_content(
            model=self.model_id, contents=prompt, config=config
        )
        
        # Capture usage
        if hasattr(response, 'usage_metadata'):
             # google-genai usage_metadata structure
             # usually: prompt_token_count, candidates_token_count, total_token_count
             u = response.usage_metadata
             p_tok = getattr(u, 'prompt_token_count', 0)
             c_tok = getattr(u, 'candidates_token_count', 0)
             global_token_tracker.update(prompt_tokens=p_tok, completion_tokens=c_tok)

        return core_types.ScoredOutput(score=1.0, output=response.text)

    except Exception as e:
        # Fallback to original if something goes wrong with our patch logic?
        # Or just re-raise. Re-raising is safer to match original behavior.
        # But for safety, if we fail to patch, we might just call original?
        # No, if we are here, we are already executing.
        
        # To match original error handling:
        # raise exceptions.InferenceRuntimeError(f'Gemini API error: {str(e)}', original=e) from e
        # checking imports... we need exceptions from langextract.core
        from langextract.core import exceptions
        raise exceptions.InferenceRuntimeError(
            f'Gemini API error: {str(e)}', original=e
        ) from e

# Apply patch
GeminiLanguageModel._process_single_prompt = _patched_process_single_prompt
# ---------------------------------------

def _configure_langextract_output():
    # Keep LangExtract output clean; opt-in to raw tqdm via env var.
    enable_tqdm = os.getenv("HTMLMINER_LANGEXTRACT_TQDM", "").lower() in ("1", "true", "yes")
    from langextract import progress as lx_progress

    # Strip ANSI color codes to avoid raw escape sequences in some terminals.
    lx_progress.BLUE = ""
    lx_progress.GREEN = ""
    lx_progress.CYAN = ""
    lx_progress.BOLD = ""
    lx_progress.RESET = ""

    if enable_tqdm:
        return

    def _wrap_progress_bar(factory):
        def _wrapper(*args, **kwargs):
            kwargs["disable"] = True
            return factory(*args, **kwargs)
        return _wrapper

    lx_progress.create_download_progress_bar = _wrap_progress_bar(
        lx_progress.create_download_progress_bar
    )
    lx_progress.create_extraction_progress_bar = _wrap_progress_bar(
        lx_progress.create_extraction_progress_bar
    )
    lx_progress.create_save_progress_bar = _wrap_progress_bar(
        lx_progress.create_save_progress_bar
    )
    lx_progress.create_load_progress_bar = _wrap_progress_bar(
        lx_progress.create_load_progress_bar
    )

_configure_langextract_output()


console = Console()

from .database import log_event

MODEL_TIERS = {
    "cheap": {
        "model_id": "gemini-2.5-flash",
        "dspy_model": "gemini/gemini-2.5-flash",
    },
    "expensive": {
        "model_id": "gemini-2.5-pro",
        "dspy_model": "gemini/gemini-2.5-pro",
    },
}

class AgenticExtractor:
    LONG_PAGE_CHAR_THRESHOLD = 15000

    def __init__(
        self,
        api_key: str,
        session_id: str = None,
        extraction_config: list = None,
        model_tier: str = "cheap",
        synthesis_top: int = 50,
    ):
        self.api_key = api_key
        self.session_id = session_id
        self.extraction_config = extraction_config
        if model_tier not in MODEL_TIERS:
            raise ValueError(f"Unknown model tier: {model_tier}")
        self.model_tier = model_tier
        if synthesis_top < 1:
            raise ValueError("synthesis_top must be >= 1")
        self.synthesis_top = synthesis_top
        self.model_id = MODEL_TIERS[model_tier]["model_id"]
        self.dspy_model = MODEL_TIERS[model_tier]["dspy_model"]
        
        # Ensure the API key is set for google-generativeai / google-genai
        if api_key:
            if "GEMINI_API_KEY" not in os.environ:
                os.environ["GEMINI_API_KEY"] = api_key
            if "LANGEXTRACT_API_KEY" not in os.environ:
                os.environ["LANGEXTRACT_API_KEY"] = api_key

        # Configure dspy
        try:
             lm = dspy.LM(model=self.dspy_model, api_key=api_key)
             dspy.settings.configure(lm=lm)
             self.dspy_lm = lm # Keep ref to track usage
        except Exception as e:
            console.print(f"[yellow]Warning: Could not configure DSpy with Gemini: {e}. Synthesis might fail if it relies on DSpy.[/yellow]")
            self.dspy_lm = None
            
        # Reset token tracker at start of session/agent init? 
        # Or just let it accumulate. Global is fine for CLI run.
        # If running multiple process commands in same script execution (unlikely), it might accumulate.
        # Let's provide a reset method.
        # global_token_tracker.reset() # Optional

    def select_top_pages(
        self,
        base_url: str,
        sitemap_urls: list[str],
        limit: int,
        status_callback=None,
    ) -> list[str]:
        candidates = self._prepare_candidate_urls(base_url, sitemap_urls)
        if not candidates:
            return [base_url]
        if limit <= 0:
            return []
        effective_limit = min(limit, len(candidates))

        if status_callback:
            status_callback(f"Selecting top {effective_limit} pages from sitemap...")

        selected = self._llm_select_pages(base_url, candidates, effective_limit)
        if selected:
            if len(selected) < effective_limit:
                remainder = [u for u in candidates if u not in selected]
                fill = self._heuristic_select_pages(
                    base_url,
                    remainder,
                    effective_limit - len(selected),
                )
                return selected + fill
            return selected

        return self._heuristic_select_pages(base_url, candidates, effective_limit)

    def select_top_pages_for_features(
        self,
        base_url: str,
        sitemap_urls: list[str],
        limit: int,
        status_callback=None,
    ) -> dict:
        candidates = self._prepare_candidate_urls(base_url, sitemap_urls)
        if not candidates:
            return {"all": [base_url], "by_feature": {}}
        if limit <= 0:
            return {"all": [], "by_feature": {}}

        effective_limit = min(limit, len(candidates))
        features = self.extraction_config or [
            {"name": "Risk", "description": "AI risks and negative impacts."},
            {"name": "Goal", "description": "High-level goals or mission statements."},
            {"name": "Method", "description": "Strategies, activities, or actions taken."},
        ]
        selected_by_feature = {}
        selected_all = []

        for feature in features:
            feature_name = feature.get("name", "Feature")
            if status_callback:
                status_callback(
                    f"Selecting top {effective_limit} pages for {feature_name}..."
                )

            selected = self._llm_select_pages(
                base_url, candidates, effective_limit, feature=feature
            )
            if selected:
                if len(selected) < effective_limit:
                    remainder = [u for u in candidates if u not in selected]
                    keywords = self._keyword_set_from_feature(feature)
                    fill = self._heuristic_select_pages(
                        base_url,
                        remainder,
                        effective_limit - len(selected),
                        keywords=keywords,
                    )
                    selected = selected + fill
            else:
                keywords = self._keyword_set_from_feature(feature)
                selected = self._heuristic_select_pages(
                    base_url, candidates, effective_limit, keywords=keywords
                )

            selected_by_feature[feature_name] = selected
            for url in selected:
                if url not in selected_all:
                    selected_all.append(url)

        if not selected_all:
            selected_all = candidates[:effective_limit]

        return {"all": selected_all, "by_feature": selected_by_feature}

    def _prepare_candidate_urls(
        self,
        base_url: str,
        sitemap_urls: list[str],
        max_candidates: int = 200,
    ) -> list[str]:
        from urllib.parse import urlparse

        def _is_same_domain(u: str) -> bool:
            try:
                return urlparse(u).netloc == urlparse(base_url).netloc
            except Exception:
                return False

        def _is_disallowed(u: str) -> bool:
            disallowed_ext = (
                ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg",
                ".mp4", ".mov", ".zip", ".rar", ".7z",
            )
            lowered = u.lower()
            return any(lowered.endswith(ext) for ext in disallowed_ext)

        seen = set()
        filtered = []
        for u in sitemap_urls:
            if not u or u in seen:
                continue
            if not _is_same_domain(u):
                continue
            if _is_disallowed(u):
                continue
            seen.add(u)
            filtered.append(u)

        if base_url not in seen:
            filtered.insert(0, base_url)

        if len(filtered) <= max_candidates:
            return filtered

        return self._heuristic_select_pages(base_url, filtered, max_candidates)

    def _heuristic_select_pages(
        self,
        base_url: str,
        candidates: list[str],
        limit: int,
        keywords: Optional[list[str]] = None,
    ) -> list[str]:
        from urllib.parse import urlparse

        keywords = keywords or self._keyword_set_from_features()

        def score(u: str) -> float:
            if u == base_url:
                return 100.0
            parsed = urlparse(u)
            path = parsed.path.lower()
            score_val = 0.0
            for kw in keywords:
                if f"/{kw}" in path:
                    score_val += 5.0
            score_val -= min(len(path), 200) / 10.0
            return score_val

        ranked = sorted(candidates, key=score, reverse=True)
        selected = []
        seen = set()
        for u in ranked:
            if u in seen:
                continue
            seen.add(u)
            selected.append(u)
            if len(selected) >= limit:
                break
        return selected

    def _llm_select_pages(
        self,
        base_url: str,
        candidates: list[str],
        limit: int,
        feature: Optional[dict] = None,
    ) -> list[str]:
        if not self.dspy_lm:
            return []

        if feature:
            features_text = f'- "{feature.get("name", "Feature")}": {feature.get("description", "")}'
        else:
            features = self.extraction_config or [
                {"name": "Risk", "description": "AI risks and negative impacts."},
                {"name": "Goal", "description": "High-level goals or mission statements."},
                {"name": "Method", "description": "Strategies, activities, or actions taken."},
            ]
            features_text = "\n".join(
                [f'- "{f["name"]}": {f.get("description", "")}' for f in features]
            )
        candidates_text = "\n".join(candidates)

        class PageSelector(dspy.Signature):
            base_url = dspy.InputField()
            features = dspy.InputField()
            candidates = dspy.InputField()
            limit = dspy.InputField()
            selection = dspy.OutputField(
                desc="JSON array of selected URLs, ordered, max length = limit"
            )

        selector = dspy.Predict(PageSelector)
        try:
            result = selector(
                base_url=base_url,
                features=features_text,
                candidates=candidates_text,
                limit=str(limit),
            )
            raw = getattr(result, "selection", "") or str(result)
            parsed = self._parse_selected_urls(raw, candidates)
            if parsed:
                return parsed[:limit]
        except Exception:
            return []

        return []

    def _keyword_set_from_features(self) -> list[str]:
        features = self.extraction_config or [
            {"name": "Risk", "description": "AI risks and negative impacts."},
            {"name": "Goal", "description": "High-level goals or mission statements."},
            {"name": "Method", "description": "Strategies, activities, or actions taken."},
        ]
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "into", "about",
            "your", "their", "they", "them", "over", "under", "across", "between",
            "within", "without", "these", "those", "will", "shall", "would", "could",
            "should", "have", "has", "had", "been", "being", "are", "was", "were",
            "not", "any", "all", "most", "many", "more", "less", "than", "such",
            "use", "using", "used", "via", "etc",
        }
        raw_terms = []
        for f in features:
            raw_terms.append(f.get("name", ""))
            raw_terms.append(f.get("description", ""))
            raw_terms.append(f.get("synthesis_topic", ""))
        text = " ".join([t for t in raw_terms if t])
        tokens = []
        current = []
        for ch in text.lower():
            if ch.isalpha():
                current.append(ch)
            else:
                if current:
                    tokens.append("".join(current))
                    current = []
        if current:
            tokens.append("".join(current))
        filtered = [t for t in tokens if len(t) > 3 and t not in stopwords]
        base_terms = ["about", "research", "mission", "team", "policy", "safety"]
        merged = list(dict.fromkeys(filtered + base_terms))
        return merged

    def _keyword_set_from_feature(self, feature: dict) -> list[str]:
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "into", "about",
            "your", "their", "they", "them", "over", "under", "across", "between",
            "within", "without", "these", "those", "will", "shall", "would", "could",
            "should", "have", "has", "had", "been", "being", "are", "was", "were",
            "not", "any", "all", "most", "many", "more", "less", "than", "such",
            "use", "using", "used", "via", "etc",
        }
        text = " ".join(
            [
                feature.get("name", ""),
                feature.get("description", ""),
                feature.get("synthesis_topic", ""),
            ]
        )
        tokens = []
        current = []
        for ch in text.lower():
            if ch.isalpha():
                current.append(ch)
            else:
                if current:
                    tokens.append("".join(current))
                    current = []
        if current:
            tokens.append("".join(current))
        filtered = [t for t in tokens if len(t) > 3 and t not in stopwords]
        base_terms = ["about", "research", "mission", "team", "policy", "safety"]
        return list(dict.fromkeys(filtered + base_terms))

    def _parse_selected_urls(self, raw: str, candidates: list[str]) -> list[str]:
        import json
        import re

        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [u for u in data if u in candidates]
        except Exception:
            pass

        urls = re.findall(r"https?://\\S+", raw)
        return [u for u in urls if u in candidates]

    def get_token_usage(self):
        """Returns aggregated token usage from LangExtract and DSpy."""
        # Get LangExtract usage
        stats = global_token_tracker.get_stats()
        
        # Add DSpy usage
        if self.dspy_lm and self.dspy_lm.history:
             # Iterate history to sum up usage?
             # dspy.LM doesn't expose a total_usage counter directly usually, we have to sum history.
             # but history might be long.
             # simpler: dspy usually prints cost/usage.
             # Let's sum up history of this instance.
             d_prompt = 0
             d_compl = 0
             
             # Note: history grows indefinitely. In a long run, this is bad. 
             # But for a CLI tool it's okay.
             for h in self.dspy_lm.history:
                 u = h.get('usage', {})
                 if u:
                     # dspy usage dict keys vary by provider, but usually:
                     # 'prompt_tokens', 'completion_tokens'
                     d_prompt += u.get('prompt_tokens', 0)
                     d_compl += u.get('completion_tokens', 0)
            
             stats["prompt_tokens"] += d_prompt
             stats["completion_tokens"] += d_compl
             stats["total_tokens"] += (d_prompt + d_compl)
             
        return stats

    def extract(
        self,
        snapshot: str,
        url: str,
        max_paragraphs: int = 3,
        status_callback=None,
        site_context: Optional[dict] = None,
        short_page_contexts: Optional[list[tuple[str, str]]] = None,
    ) -> dict:
        """
        Extracts Risk, Goals, and Methods from the snapshot.
        """
        short_page_contexts = short_page_contexts or []
        
        # 1. Check size
        size_limit_chars = 2_000_000
        
        if len(snapshot) > size_limit_chars:
            console.print(f"[bold yellow]Snapshot for {url} is very large ({len(snapshot)} chars). Truncating...[/bold yellow]")
            snapshot = snapshot[:size_limit_chars]
        
        # 2. Reformulation Loop - Dynamic generation
        if self.extraction_config:
            features_text = "\n".join([f'- "{f["name"]}": {f["description"]}' for f in self.extraction_config])
            prompt_main = textwrap.dedent(f"""
                Extract the following entities from the text:
                {features_text}
                
                Extract complete sentences or paragraphs that are conceptually relevant to each entity.
                Include surrounding context to capture the full meaning. Prefer longer, contextual excerpts over short phrases.
            """)
            prompts = [prompt_main]
        else:
             # Fallback
             prompts = [
                textwrap.dedent("""
                    Extract the following entities from the text:
                    - "Risk": Any mentioned risks, dangers, or negative impacts of AI development.
                    - "Goal": High-level goals, missions, or objectives (e.g., "AI alignment", "AGI").
                    - "Method": Strategies, activities, or actions taken to achieve the goals (e.g., "research", "grantmaking", "policy work").
                    
                    Extract complete sentences or paragraphs that are conceptually relevant to each entity.
                    Include surrounding context to capture the full meaning. Prefer longer, contextual excerpts over short phrases.
                """)
            ]

        # Use examples with longer, contextual extractions
        examples = [
            lx.data.ExampleData(
                text="The institute aims to ensure AI systems are aligned with human values. We believe this is one of the most important challenges of our time. We conduct technical research on robustness and interpretability, and we advocate for safety regulations at the policy level. A major concern we address is power-seeking behavior in advanced models, which could lead to unintended consequences if not properly controlled.",
                extractions=[
                    lx.data.Extraction(extraction_class="Goal", extraction_text="The institute aims to ensure AI systems are aligned with human values. We believe this is one of the most important challenges of our time."),
                    lx.data.Extraction(extraction_class="Method", extraction_text="We conduct technical research on robustness and interpretability, and we advocate for safety regulations at the policy level."),
                    lx.data.Extraction(extraction_class="Risk", extraction_text="A major concern we address is power-seeking behavior in advanced models, which could lead to unintended consequences if not properly controlled."),
                ]
            )
        ]

        all_extractions = []
        
        for i, prompt in enumerate(prompts):
            try:
                if status_callback:
                    status_callback(f"Extracting raw snippets (Attempt {i+1})...")
                console.print(f"[dim]Attempt {i+1} with prompt reformulation...[/dim]")
                if self.session_id:
                    log_event(self.session_id, "agent", "INFO", f"Extraction attempt {i+1} for {url}", {"prompt_seed": prompt[:100]})
                
                result = lx.extract(
                    text_or_documents=snapshot,
                    prompt_description=prompt,
                    examples=examples,
                    model_id=self.model_id,
                    api_key=self.api_key,
                )
                
                current_extractions = getattr(result, "extractions", [])
                if current_extractions:
                    all_extractions.extend(current_extractions)
                    if status_callback:
                        status_callback(f"Found {len(all_extractions)} raw snippets...")
                    if len(current_extractions) > 0:
                        break
            except Exception as e:
                console.print(f"[red]Error during extraction attempt {i+1}: {e}[/red]")

        if status_callback:
            status_callback(f"Grouping {len(all_extractions)} snippets by feature...")

        if self.extraction_config:
            features = self.extraction_config
        else:
            features = [
                {"name": "Risk", "description": "Any mentioned risks, dangers, or negative impacts of AI development."},
                {"name": "Goal", "description": "High-level goals, missions, or objectives (e.g., 'AI alignment', 'AGI')."},
                {"name": "Method", "description": "Strategies, activities, or actions taken to achieve the goals (e.g., 'research', 'grantmaking', 'policy work')."}
            ]

        # 3. Synthesis Feature-by-Feature
        result_data = {
            "URL": url,
            "Raw_Extractions": len(all_extractions)
        }
        
        # Parse snapshot to build page mapping for source tracking
        import re
        page_sections = []
        # Match patterns like "# Page 1: https://example.com/about"
        page_pattern = re.compile(r'^# Page \d+: (.+)$', re.MULTILINE)
        matches = list(page_pattern.finditer(snapshot))
        
        if matches:
            for i, match in enumerate(matches):
                page_url = match.group(1).strip()
                start_pos = match.end()
                end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(snapshot)
                page_content = snapshot[start_pos:end_pos]
                page_sections.append({"url": page_url, "content": page_content})
        else:
            # Single page, use the main URL
            page_sections.append({"url": url, "content": snapshot})
        
        def find_source_page(text: str) -> str:
            """Find which page section contains the extraction text."""
            for section in page_sections:
                if text in section["content"]:
                    return section["url"]
            return url  # Default to main URL if not found
        
        grouped_extractions = {}
        seen_texts = set()  # For deduplication
        MIN_WORD_COUNT = 5  # Minimum words for an extraction to be meaningful
        
        for e in all_extractions:
            # Skip empty strings and 'null' values
            text = e.extraction_text
            if not text or text.strip() == "" or text.strip().lower() == "null":
                continue
            
            text = text.strip()
            
            # Skip short uninformative snippets (less than MIN_WORD_COUNT words)
            word_count = len(text.split())
            if word_count < MIN_WORD_COUNT:
                continue
            
            # Skip duplicates
            text_lower = text.lower()
            if text_lower in seen_texts:
                continue
            seen_texts.add(text_lower)
                
            cls = e.extraction_class
            if cls not in grouped_extractions:
                grouped_extractions[cls] = []
            # Store as structured object with text and source
            source = find_source_page(text)
            grouped_extractions[cls].append({
                "text": text,
                "source": source
            })
            
        short_context_blob = self._format_short_page_contexts(short_page_contexts)

        for idx, feature in enumerate(features):
            f_name = feature["name"]
            # synthesis topic matches feature usually, or explicitly set
            f_topic = feature.get("synthesis_topic", f_name)
            
            snippets = grouped_extractions.get(f_name, [])
            
            if status_callback:
                status_callback(f"Synthesizing {f_name} ({idx+1}/{len(features)}, {len(snippets)} snippets)...")
            
            # Limit synthesis context to the top N longest extractions.
            snippets_for_synthesis = sorted(
                snippets, key=lambda s: len(s["text"]), reverse=True
            )[: self.synthesis_top]
            text_blob = "\n".join([s["text"] for s in snippets_for_synthesis])
            combined_context = "\n\n".join(
                [chunk for chunk in [text_blob, short_context_blob] if chunk]
            )

            if not combined_context:
                result_data[f_name] = "Not mentioned."
                result_data[f"{f_name}_Raw"] = []
                result_data[f"{f_name}_Count"] = 0
            else:
                result_data[f_name] = self.synthesize_feature(
                    f_name,
                    f_topic,
                    combined_context,
                    max_paragraphs,
                    site_context=site_context,
                )
                result_data[f"{f_name}_Raw"] = snippets  # List of {text, source} objects
                result_data[f"{f_name}_Count"] = len(snippets)

        return result_data

    def extract_from_pages(
        self,
        pages: list[tuple[str, str]],
        url: str,
        max_paragraphs: int = 3,
        status_callback=None,
        site_context: Optional[dict] = None,
    ) -> dict:
        long_pages = []
        short_pages = []
        for page_url, content in pages:
            if content and len(content) >= self.LONG_PAGE_CHAR_THRESHOLD:
                long_pages.append((page_url, content))
            else:
                short_pages.append((page_url, content))

        if not long_pages:
            return self._extract_from_short_pages(
                short_pages,
                url,
                max_paragraphs=max_paragraphs,
                status_callback=status_callback,
                site_context=site_context,
            )

        combined = self._combine_pages(long_pages)
        return self.extract(
            combined,
            url,
            max_paragraphs=max_paragraphs,
            status_callback=status_callback,
            site_context=site_context,
            short_page_contexts=short_pages,
        )

    def _extract_from_short_pages(
        self,
        pages: list[tuple[str, str]],
        url: str,
        max_paragraphs: int = 3,
        status_callback=None,
        site_context: Optional[dict] = None,
    ) -> dict:
        if self.extraction_config:
            features = self.extraction_config
        else:
            features = [
                {"name": "Risk", "description": "Any mentioned risks, dangers, or negative impacts of AI development."},
                {"name": "Goal", "description": "High-level goals, missions, or objectives (e.g., 'AI alignment', 'AGI')."},
                {"name": "Method", "description": "Strategies, activities, or actions taken to achieve the goals (e.g., 'research', 'grantmaking', 'policy work')."}
            ]

        result_data = {
            "URL": url,
            "Raw_Extractions": 0,
        }
        short_context_blob = self._format_short_page_contexts(pages)

        for idx, feature in enumerate(features):
            f_name = feature["name"]
            f_topic = feature.get("synthesis_topic", f_name)
            if status_callback:
                status_callback(f"Synthesizing {f_name} ({idx+1}/{len(features)}) from short pages...")

            if not short_context_blob:
                result_data[f_name] = "Not mentioned."
                result_data[f"{f_name}_Raw"] = []
                result_data[f"{f_name}_Count"] = 0
            else:
                result_data[f_name] = self.synthesize_feature(
                    f_name,
                    f_topic,
                    short_context_blob,
                    max_paragraphs,
                    site_context=site_context,
                )
                result_data[f"{f_name}_Raw"] = []
                result_data[f"{f_name}_Count"] = 0

        return result_data

    def _combine_pages(self, pages: list[tuple[str, str]]) -> str:
        combined = []
        for i, (page_url, page_content) in enumerate(pages):
            combined.append(f"# Page {i+1}: {page_url}\n\n{page_content}\n")
        return "\n\n---\n\n".join(combined)

    def _format_short_page_contexts(self, pages: list[tuple[str, str]]) -> str:
        if not pages:
            return ""
        parts = []
        for page_url, content in pages:
            if not content:
                continue
            parts.append(f"# Page: {page_url}\n\n{content}")
        return "\n\n---\n\n".join(parts)

    def synthesize_feature(
        self,
        feature_name: str,
        topic_desc: str,
        extracted_data: str,
        max_paragraphs: int = 1,
        site_context: Optional[dict] = None,
    ) -> str:
        """
        Uses DSpy/LLM to synthesize the extractions for a single feature.
        """
        if site_context:
            extracted_data = f"{extracted_data}\n\n{self._format_site_context(site_context)}"

        try:
            class FeatureSummarizer(dspy.Signature):
                """Summarize the extracted information for a specific feature."""
                context = dspy.InputField()
                summary = dspy.OutputField(desc=f"Concise summary of {feature_name}")

            summarize_prog = dspy.Predict(FeatureSummarizer)
            result = summarize_prog(context=extracted_data)
            return result.summary
        except Exception as e:
            console.print(f"[red]Synthesis failed for {feature_name}: {e}[/red]")
            return f"Error synthesizing {feature_name}."

    def _format_site_context(self, site_context: dict) -> str:
        sitemap_urls = site_context.get("sitemap_urls") or []
        scraped_paths = site_context.get("scraped_paths") or []
        scraped_urls = site_context.get("scraped_urls") or []
        selected_urls_by_feature = site_context.get("selected_urls_by_feature") or {}
        max_items = site_context.get("max_items", 50)

        lines = ["Site context (sitemap + scraped paths for deeper review if needed):"]
        if sitemap_urls:
            lines.append("Sitemap URLs:")
            lines.extend([f"- {u}" for u in sitemap_urls[:max_items]])
            if len(sitemap_urls) > max_items:
                lines.append(f"- ... ({len(sitemap_urls) - max_items} more)")
        if scraped_paths:
            lines.append("Scraped paths:")
            lines.extend([f"- {p}" for p in scraped_paths[:max_items]])
            if len(scraped_paths) > max_items:
                lines.append(f"- ... ({len(scraped_paths) - max_items} more)")
        if scraped_urls and not scraped_paths:
            lines.append("Scraped URLs:")
            lines.extend([f"- {u}" for u in scraped_urls[:max_items]])
            if len(scraped_urls) > max_items:
                lines.append(f"- ... ({len(scraped_urls) - max_items} more)")
        if selected_urls_by_feature:
            lines.append("Selected URLs by feature:")
            for feature_name, urls in selected_urls_by_feature.items():
                lines.append(f"{feature_name}:")
                lines.extend([f"- {u}" for u in urls[:max_items]])
                if len(urls) > max_items:
                    lines.append(f"- ... ({len(urls) - max_items} more)")
        return "\n".join(lines)
