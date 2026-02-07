import os
import pandas as pd
from rich.console import Console
from rich.table import Table
from typing import List, Dict, Optional
from datetime import datetime

console = Console()

def save_results(results: List[Dict], output_file: str, metadata: Optional[Dict] = None):
    """
    Saves extraction results to a file.
    For JSON: includes metadata section with dates, config, and parameters.
    For CSV: saves only the results data.
    """
    if not results:
        console.print("[yellow]No results to save.[/yellow]")
        return

    df = pd.DataFrame(results)
    
    if output_file.endswith(".json"):
        import json
        # Build output with metadata
        output_data = {
            "metadata": metadata or {},
            "results": results
        }
        # Add timestamp if not present
        if "timestamp" not in output_data["metadata"]:
            output_data["metadata"]["timestamp"] = datetime.now().isoformat()

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4, default=str)
    else:
        df.to_csv(output_file, index=False)

    console.print(f"[green]Results saved to {os.path.abspath(output_file)}[/green]")

def save_summary_csv(results: List[Dict], output_file: str, show_full_path: bool = False):
    """
    Saves a summary CSV with URL, feature columns, and a counts column.
    This matches the CLI table format.
    """
    if not results:
        console.print("[yellow]No results to save.[/yellow]")
        return

    # Extract feature names (excluding special fields)
    known_fields = {"URL", "Raw_Extraction_By_Page"}
    feature_names = []
    for res in results:
        for k in res.keys():
            if k not in known_fields and k not in feature_names and not k.endswith("_Raw") and not k.endswith("_Count"):
                feature_names.append(k)

    # Build CSV data
    csv_data = []
    for res in results:
        row = {"URL": res.get("URL", "N/A")}

        # Add feature columns
        for feature in feature_names:
            row[feature] = res.get(feature, "-")

        # Build counts column
        counts = []
        for feature in feature_names:
            count = res.get(f"{feature}_Count", 0)
            if count > 0:
                counts.append(f"{feature}: {count}")
        row["Counts"] = ", ".join(counts) if counts else "Synthesized"

        csv_data.append(row)

    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    display_path = os.path.abspath(output_file) if show_full_path else output_file
    console.print(f"[green]Summary saved to {display_path}[/green]")

def display_results(results: List[Dict]):
    """
    Displays the results in a Rich table with dynamic columns.
    """
    if not results:
        return

    table = Table(title="Agentic Extraction Results")

    # Fixed columns
    table.add_column("URL", style="cyan", no_wrap=True)

    # Dynamic columns
    # We'll take the keys from the first result, excluding known fixed fields
    known_fields = {"URL", "Raw_Extraction_By_Page", "Counts"}

    # Gather all potential keys from all results to be safe, or just first one
    all_keys = []
    for res in results:
        for k in res.keys():
            if k not in known_fields and k not in all_keys and not k.endswith("_Raw") and not k.endswith("_Count"):
                all_keys.append(k)

    # Sort or keep order? 'Risk', 'Goal', 'Method' usually come in that order if config preserves it.
    # We'll trust the insertion order (Python dicts preserve order).

    for key in all_keys:
        table.add_column(key, style="magenta")

    table.add_column("Counts", justify="right")

    for res in results:
        row_values = [res.get("URL", "N/A")]

        for key in all_keys:
            val = res.get(key, "-")
            # Ensure val is a string
            if not isinstance(val, str):
                val = str(val) if val is not None else "-"
            # Truncate for display
            display_val = (val[:200] + "...") if len(val) > 200 else val
            row_values.append(display_val)

        # Build counts string - handle both old format (_Count fields) and new format (Counts dict)
        counts_dict = res.get("Counts", {})
        counts = []
        
        if isinstance(counts_dict, dict):
            # New graph workflow format: Counts is a dict
            for feature, count in counts_dict.items():
                if count and count > 0:
                    counts.append(f"{feature}: {count}")
        else:
            # Old format: individual _Count fields
            for key in all_keys:
                count = res.get(f"{key}_Count", 0)
                if count > 0:
                    counts.append(f"{key}: {count}")

        counts_str = ", ".join(counts) if counts else "Synthesized"
        row_values.append(counts_str)

        table.add_row(*row_values)

    console.print(table)
