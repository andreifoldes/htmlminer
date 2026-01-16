import json
import sys
import os

def validate_results(filepath):
    """
    Validates that the output JSON contains non-empty results for key features.
    """
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from {filepath}: {e}")
        return False

    # Handle both list (legacy/simple) and dict (new) formats
    if isinstance(data, dict) and 'results' in data:
        results = data['results']
    elif isinstance(data, list):
        results = data
    else:
        print(f"Error: Unexpected JSON structure in {filepath}. Expected dict with 'results' key or a list.")
        return False

    if not results:
        print(f"Error: No results found in {filepath}.")
        return False

    required_keys = ['Risk', 'Goal', 'Method']
    success = True

    for i, item in enumerate(results):
        url = item.get('URL', 'Unknown URL')
        print(f"Validating result {i+1} for URL: {url}")
        
        for key in required_keys:
            value = item.get(key)
            if not value or not isinstance(value, str) or not value.strip():
                print(f"  [FAILURE] Missing or empty '{key}'")
                success = False
            else:
                print(f"  [OK] '{key}' is present (length: {len(value)})")

    if success:
        print(f"Validation PASSED for {filepath}")
        return True
    else:
        print(f"Validation FAILED for {filepath}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_ci_results.py <path_to_results.json>")
        sys.exit(1)

    filepath = sys.argv[1]
    if validate_results(filepath):
        sys.exit(0)
    else:
        sys.exit(1)
