"""
Extract triplets from a specific row and DB column.
Usage:
    python extract_case.py --csv data.csv --row 382 --col generated_db_1
"""
import ast, csv, sys, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--row', type=int, required=True, help='Row index (0-based, excluding header)')
    parser.add_argument('--col', default='generated_db_1', help='Column name')
    args = parser.parse_args()

    with open(args.csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        
        col_idx = header.index(args.col)
        prompt_idx = header.index('phase2_prompt')
        answer_idx = header.index('ground_truth_answer')
        
        for i, row in enumerate(reader):
            if i == args.row:
                print(f"=== Row {i} | {args.col} ===")
                print(f"\nQuestion: {row[prompt_idx].strip()}")
                print(f"Answer: {row[answer_idx].strip()}")
                print(f"\n--- Triplets ---")
                raw = row[col_idx].strip()
                triplets = ast.literal_eval(raw)
                for t in triplets:
                    print(t)
                print(f"\n--- Raw (copy this) ---")
                print(raw)
                return
        print(f"Row {args.row} not found!")

if __name__ == '__main__':
    main()