import pandas as pd
import argparse

def extract_rows(input_file, output_file, num_rows):
    """
    Extract the first X rows from a CSV file and save to a new CSV file.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
        num_rows (int): Number of rows to extract
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Extract the first X rows
        df_subset = df.head(num_rows)
        
        # Save to new CSV file
        df_subset.to_csv(output_file, index=False)
        print(f"Successfully extracted {num_rows} rows from {input_file}")
        print(f"Saved to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract first X rows from a CSV file')
    parser.add_argument('--input_file', required=True, help='Path to the input CSV file')
    parser.add_argument('--output_file', required=True, help='Path to save the output CSV file')
    parser.add_argument('--num_rows', type=int, required=True, help='Number of rows to extract')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the extraction
    extract_rows(args.input_file, args.output_file, args.num_rows) 