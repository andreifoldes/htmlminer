import pandas as pd
from rich.console import Console
from rich.table import Table
from typing import List, Dict

console = Console()

def save_results(results: List[Dict], output_file: str):
    """
    Saves extraction results to a CSV file.
    """
    if not results:
        console.print("[yellow]No results to save.[/yellow]")
        return

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    console.print(f"[green]Results saved to {output_file}[/green]")

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
    known_fields = {"URL", "Raw_Extractions"}
    
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
            # Truncate for display
            display_val = (val[:200] + "...") if len(val) > 200 else val
            row_values.append(display_val)
            
        # Build counts string
        counts = []
        for key in all_keys:
             count = res.get(f"{key}_Count", 0)
             if count > 0:
                 counts.append(f"{key}: {count}")
        
        counts_str = ", ".join(counts) if counts else "0"
        row_values.append(counts_str)
        
        table.add_row(*row_values)

    console.print(table)
