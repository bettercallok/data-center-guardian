"""
extract_columns.py — compile the "Failure Five" SMART columns from a folder of
Backblaze daily CSVs into a single CSV (pure stdlib, low memory).

Usage:
  python extract_columns.py --input-dir ~/Downloads/data_Q4_2025 \
                            --output compiled_smart_data.csv
"""
import argparse
import csv
import glob
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="Folder containing Backblaze daily CSV files")
    parser.add_argument("--output", default="compiled_smart_data.csv", help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    input_dir = os.path.expanduser(args.input_dir)
    output_file = os.path.expanduser(args.output)

    target_columns = [
        'date',
        'serial_number',
        'model',
        'failure',
        'smart_5_raw',
        'smart_187_raw',
        'smart_188_raw',
        'smart_197_raw',
        'smart_198_raw'
    ]

    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    csv_files.sort()
    
    print(f"Found {len(csv_files)} CSV files to process.")

    with open(output_file, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(target_columns)
        
        for idx, file in enumerate(csv_files):
            # Print progress every 10 files
            if idx % 10 == 0:
                print(f"Processing file {idx+1}/{len(csv_files)}: {os.path.basename(file)}")
                
            with open(file, 'r', newline='') as in_f:
                reader = csv.reader(in_f)
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                    
                # Find indices for this specific file
                try:
                    indices = [header.index(col) for col in target_columns]
                except ValueError as e:
                    print(f"Skipping {os.path.basename(file)} due to missing column: {e}")
                    continue
                
                # Write rows
                for row in reader:
                    # Only write rows that have enough columns
                    if len(row) > max(indices):
                        writer.writerow([row[i] for i in indices])
                    else:
                        writer.writerow([row[i] if i < len(row) else '' for i in indices])

    elapsed = time.time() - start_time
    print(f"\nData extraction complete in {elapsed:.2f} seconds.")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()
