#!/usr/bin/env python3
"""
Script to create a research paper style results table from JSON files.
Usage: python summarize_results.py <folder_path>
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


def load_json_files(folder_path: str) -> List[Dict[str, Any]]:
    """Load all JSON files from the specified folder."""
    results = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return results
    
    json_files = sorted(folder.glob("*.json"))
    
    if not json_files:
        print(f"Warning: No JSON files found in '{folder_path}'")
        return results
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
    
    return results


def extract_setting_name(preds_path: str) -> str:
    """Extract a clean setting name from the preds_path."""
    # Example: "./output_eval/generations/eval_hotpotqa_train_Qwen2.5-3B-SFT_ep5_bsz48_th-1-ckpt600_n100_i90347.json"
    # Extract the meaningful part
    filename = Path(preds_path).stem
    
    # Remove common prefixes
    if filename.startswith('eval_'):
        filename = filename[5:]
    
    return filename


def create_results_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a clean results table."""
    rows = []
    
    for result in results:
        metrics = result.get('metrics', {})
        meta = result.get('meta', {})
        
        row = {
            'Setting': extract_setting_name(meta.get('preds_path', 'unknown')),
            'Split': meta.get('split', 'unknown'),
            'Count': metrics.get('count', 0),
            'EM': f"{metrics.get('em', 0.0) * 100:.2f}",
            'F1': f"{metrics.get('f1', 0.0) * 100:.2f}",
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by F1 score (descending)
    df['_f1_sort'] = df['F1'].astype(float)
    df = df.sort_values('_f1_sort', ascending=False).drop('_f1_sort', axis=1)
    
    return df


def print_latex_table(df: pd.DataFrame):
    """Print LaTeX formatted table."""
    print("\n% LaTeX Table")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\begin{tabular}{l|c|c|c|c}")
    print("\\toprule")
    print("Setting & Split & Count & EM & F1 \\\\")
    print("\\midrule")
    
    for _, row in df.iterrows():
        print(f"{row['Setting']} & {row['Split']} & {row['Count']} & {row['EM']} & {row['F1']} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Experimental results on different settings.}")
    print("\\label{tab:results}")
    print("\\end{table}")


def print_markdown_table(df: pd.DataFrame):
    """Print Markdown formatted table."""
    print("\n# Results Table\n")
    print(df.to_markdown(index=False))


def print_simple_table(df: pd.DataFrame):
    """Print simple ASCII table."""
    print("\n" + "="*100)
    print("RESULTS TABLE")
    print("="*100 + "\n")
    print(df.to_string(index=False))
    print("\n" + "="*100)


def main():
    # Get folder path from command line or use default
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("Enter folder path (or press Enter for current directory): ").strip()
        if not folder_path:
            folder_path = "."
    
    # Get output format
    format_type = sys.argv[2] if len(sys.argv) > 2 else 'simple'
    
    # Load and process results
    print(f"Loading JSON files from: {folder_path}")
    results = load_json_files(folder_path)
    
    if not results:
        print("No results to process")
        return
    
    print(f"Loaded {len(results)} files successfully\n")
    
    # Create results table
    df = create_results_table(results)
    
    # Print in requested format
    if format_type == 'latex':
        print_latex_table(df)
    elif format_type == 'markdown':
        print_markdown_table(df)
    else:
        print_simple_table(df)
    
    # Save options
    print("\n" + "="*100)
    print("Save options:")
    print("  CSV: python script.py <folder> csv")
    print("  Markdown: python script.py <folder> markdown")
    print("  LaTeX: python script.py <folder> latex")
    print("="*100)
    
    # Save to CSV if requested
    if format_type == 'csv' or input("\nSave to CSV? (y/n): ").strip().lower() == 'y':
        output_file = "results_table.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()