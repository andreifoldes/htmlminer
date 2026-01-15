import os
import langextract as lx
import textwrap
import dspy
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
    def __init__(
        self,
        api_key: str,
        session_id: str = None,
        extraction_config: list = None,
        model_tier: str = "cheap",
    ):
        self.api_key = api_key
        self.session_id = session_id
        self.extraction_config = extraction_config
        if model_tier not in MODEL_TIERS:
            raise ValueError(f"Unknown model tier: {model_tier}")
        self.model_tier = model_tier
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

    def extract(self, snapshot: str, url: str, max_paragraphs: int = 3) -> dict:
        """
        Extracts Risk, Goals, and Methods from the snapshot.
        """
        
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
                
                Be precise and use exact text where possible.
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
                    
                    Be precise and use exact text where possible.
                """)
            ]

        # Use a generic example
        examples = [
            lx.data.ExampleData(
                text="The institute aims to ensure AI systems are aligned with human values. We conduct technical research on robustness and advocate for safety regulations. A major concern is power-seeking behavior in advanced models.",
                extractions=[
                    lx.data.Extraction(extraction_class="Goal", extraction_text="ensure AI systems are aligned with human values"),
                    lx.data.Extraction(extraction_class="Method", extraction_text="conduct technical research on robustness"),
                    lx.data.Extraction(extraction_class="Method", extraction_text="advocate for safety regulations"),
                    lx.data.Extraction(extraction_class="Risk", extraction_text="power-seeking behavior in advanced models"),
                ]
            )
        ]

        all_extractions = []
        
        for i, prompt in enumerate(prompts):
            try:
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
                
                # Check consistency / emptiness
                current_extractions = getattr(result, "extractions", [])
                if current_extractions:
                    all_extractions.extend(current_extractions)
                    if len(current_extractions) > 0:
                        break
            except Exception as e:
                console.print(f"[red]Error during extraction attempt {i+1}: {e}[/red]")

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
        
        grouped_extractions = {}
        for e in all_extractions:
            cls = e.extraction_class
            if cls not in grouped_extractions:
                grouped_extractions[cls] = []
            grouped_extractions[cls].append(e.extraction_text)
            
        for feature in features:
            f_name = feature["name"]
            # synthesis topic matches feature usually, or explicitly set
            f_topic = feature.get("synthesis_topic", f_name)
            
            snippets = grouped_extractions.get(f_name, [])
            
            if not snippets:
                result_data[f_name] = "Not mentioned."
                result_data[f"{f_name}_Raw"] = ""
                result_data[f"{f_name}_Count"] = 0
            else:
                text_blob = "\n".join(snippets)
                result_data[f_name] = self.synthesize_feature(f_name, f_topic, text_blob, max_paragraphs)
                result_data[f"{f_name}_Raw"] = " | ".join(snippets)
                result_data[f"{f_name}_Count"] = len(snippets)

        return result_data

    def synthesize_feature(self, feature_name: str, topic_desc: str, extracted_data: str, max_paragraphs: int = 1) -> str:
        """
        Uses DSpy/LLM to synthesize the extractions for a single feature.
        """
        prompt = f"""
        Based on the following extracted statements regarding '{feature_name}':
        
        {extracted_data}
        
        Summarize the {topic_desc} in {max_paragraphs} paragraph(s).
        Be concise.
        """
        
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
