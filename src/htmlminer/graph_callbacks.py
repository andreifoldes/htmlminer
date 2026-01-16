"""
HTMLMiner Graph Callbacks

LangChain callback handlers for timing and token tracking.
"""

import time
from typing import Any, Dict, List, Optional, Union
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage

from .database import log_step_timing


class TimingTokenCallback(BaseCallbackHandler):
    """
    Callback handler that tracks timing and token usage for LLM calls.
    
    Usage:
        callback = TimingTokenCallback(session_id="...")
        llm.invoke(prompt, config={"callbacks": [callback]})
        print(callback.get_totals())
    """
    
    def __init__(self, session_id: str = None, step_name: str = "llm_call"):
        super().__init__()
        self.session_id = session_id
        self.step_name = step_name
        self.start_time: Optional[float] = None
        
        # Accumulated totals
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.total_duration = 0.0
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts processing."""
        self.start_time = time.time()
        self.last_prompts = prompts
    
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        """Called when chat model starts processing."""
        self.start_time = time.time()
        # Flatten messages to string for token estimation
        full_text = ""
        for sublist in messages:
            for msg in sublist:
                str_content = ""
                if isinstance(msg.content, str):
                    str_content = msg.content
                elif isinstance(msg.content, list):
                     for part in msg.content:
                         if isinstance(part, str):
                             str_content += part
                         elif isinstance(part, dict) and 'text' in part:
                             str_content += part['text']
                full_text += str_content
        self.last_prompts = [full_text]
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM finishes. Extracts token usage."""
        duration = time.time() - self.start_time if self.start_time else 0.0
        self.total_duration += duration
        self.total_calls += 1
        
        # Extract token usage from response
        prompt_tokens = 0
        completion_tokens = 0
        
        if response.llm_output:
            usage = response.llm_output.get("usage", {})
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        
        # Also check generation info
        for generation_list in response.generations:
            for generation in generation_list:
                if hasattr(generation, "generation_info") and generation.generation_info:
                    usage = generation.generation_info.get("usage_metadata", {})
                    if usage:
                        prompt_tokens = max(prompt_tokens, usage.get("prompt_token_count", 0))
                        completion_tokens = max(completion_tokens, usage.get("candidates_token_count", 0))
        
        # Fallback to estimation if no tokens found
        if prompt_tokens == 0:
             # Estimate 1 token ~= 4 chars
             prompts_text = "".join(getattr(self, "last_prompts", []))
             if prompts_text:
                 prompt_tokens = len(prompts_text) // 4

        # If we still have 0, try to estimate from the response text at least for completion
        if completion_tokens == 0:
             completion_text = ""
             for generation_list in response.generations:
                 for generation in generation_list:
                     completion_text += generation.text
             completion_tokens = len(completion_text) // 4
        
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        
        # Log to database
        if self.session_id:
            log_step_timing(
                session_id=self.session_id,
                step_name=self.step_name,
                duration_seconds=duration,
                details={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM errors."""
        duration = time.time() - self.start_time if self.start_time else 0.0
        if self.session_id:
            log_step_timing(
                session_id=self.session_id,
                step_name=f"{self.step_name}_error",
                duration_seconds=duration,
                details={"error": str(error)}
            )
    
    def get_totals(self) -> dict:
        """Get accumulated totals."""
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_calls": self.total_calls,
            "total_duration_seconds": self.total_duration,
        }
    
    def reset(self) -> None:
        """Reset all counters."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.total_duration = 0.0


class StepTimingWrapper:
    """
    Context manager for timing workflow steps.
    
    Usage:
        with StepTimingWrapper("fetch_sitemap", session_id) as timer:
            # do work
            timer.set_details({"url_count": 100})
    """
    
    def __init__(self, step_name: str, session_id: str = None, url: str = None, status_callback: callable = None):
        self.step_name = step_name
        self.session_id = session_id
        self.url = url
        self.status_callback = status_callback
        self.start_time: Optional[float] = None
        self.details: dict = {}
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type:
            self.details["error"] = str(exc_val)
        
        if self.session_id:
            log_step_timing(
                session_id=self.session_id,
                step_name=self.step_name,
                duration_seconds=duration,
                url=self.url,
                details=self.details if self.details else None,
            )
            
        if self.status_callback and not exc_type:
            # Report successful completion back to UI
            # We use a special prefix that the UI knows to print permanently
            self.status_callback(f"✓ {self.step_name} completed in {duration:.2f}s")
        
        return False  # Don't suppress exceptions
    
    def set_details(self, details: dict) -> None:
        """Set additional details to log."""
        self.details.update(details)
