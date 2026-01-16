
import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
import langextract as lx
from langextract.core.data import ExampleData, Extraction

# Load env vars
load_dotenv(find_dotenv())

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

# Mock callback to print what it sees
class DebugCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"DEBUG: on_llm_start")
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        print(f"DEBUG: on_llm_end")
        if response.llm_output:
             print(f"DEBUG: llm_output: {response.llm_output}")
        if response.generations:
            print(f"DEBUG: Generations count: {len(response.generations)}")
            if len(response.generations) > 0 and len(response.generations[0]) > 0:
                 print(f"DEBUG: Gen[0][0] info: {response.generations[0][0].generation_info}")

class TestSchema(BaseModel):
    summary: str = Field(description="A summary")

def list_models():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("No API Key")
        return None
    genai.configure(api_key=api_key)
    print("Available Models:")
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        print(models)
        return models[0] if models else "models/gemini-1.5-flash"
    except Exception as e:
        print(f"List models failed: {e}")
        return "gemini-1.5-flash"

def test_langchain_tokens(model_name):
    # Remove 'models/' prefix if present for langchain
    model_name = model_name.replace("models/", "")
    print(f"\n=== Testing LangChain Tokens with {model_name} ===")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    callback_ctor = DebugCallback()
    
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
        callbacks=[callback_ctor] # Attach to constructor
    )
    
    # Test 2: Structured Output
    print("\n--- Structured Output (Callback in Invoke) ---")
    structured_llm = llm.with_structured_output(TestSchema)
    
    callback_invoke = DebugCallback()
    
    try:
        # Note: In graph_nodes.py, callbacks are passed in config
        result = structured_llm.invoke(
            [HumanMessage(content="Explain quantum mechanics in 5 words.")],
            config={"callbacks": [callback_invoke]}
        )
        print(f"Result type: {type(result)}")
    except Exception as e:
        print(f"Structured invoke failed: {e}")

def test_langextract_tokens():
    print("\n=== Testing LangExtract Tokens ===")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return

    # Using a known model for langextract
    model_id = "gemini/gemini-1.5-flash"
    
    content = "The quick brown fox jumps over the lazy dog."
    desc = "animal actions"
    examples = [ExampleData(text="Bird flies", extractions=[Extraction(extraction_class="action", extraction_text="flies")])]
    
    try:
        annotated_doc = lx.extract(
            content,
            prompt_description=desc,
            model_id=model_id,
            api_key=api_key,
            examples=examples
        )
        print(f"LangExtract Result Type: {type(annotated_doc)}")
        print(f"Result attributes: {annotated_doc.__dict__.keys()}")
        
    except Exception as e:
        print(f"LangExtract failed: {e}")

if __name__ == "__main__":
    model = list_models()
    if model:
        test_langchain_tokens(model)
    test_langextract_tokens()
