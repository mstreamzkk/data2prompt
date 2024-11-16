import pandas as pd
import random
import argparse
import numpy as np


def simplify_dtype(dtype):
    """Convert numpy/pandas dtypes to simple primitive type names"""
    dtype_str = str(dtype)
    
    # Numeric types
    if 'int' in dtype_str:
        return 'integer'
    if 'float' in dtype_str:
        return 'decimal'
    
    # String types
    if dtype_str in ('object', 'string', 'string[python]'):
        # Check if actually a list/dict type
        return 'string'
    
    # Boolean
    if dtype_str == 'bool':
        return 'boolean'
    
    # Datetime
    if 'datetime' in dtype_str:
        return 'datetime'
    
    # Complex types
    if 'complex' in dtype_str:
        return 'complex'
    
    # Arrays/Lists
    if isinstance(dtype, np.ndarray) or 'array' in dtype_str:
        return 'list'
    
    # Keep as object if can't determine simpler type
    return 'object'

def csv_to_text_prompt(csv_path, max_rows=None, sampling_type='sequential', segment=0):
    """
    Convert CSV to text representation with different sampling options
    
    Args:
        csv_path (str): Path to CSV file
        max_rows (int): Maximum number of rows to include, None for all rows
        sampling_type (str): 'sequential', 'random' or 'segment'
        segment (int): Segment number to return when sampling_type='segment'
        
    Returns:
        str: Text representation of CSV data
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Get headers and data types
    headers = list(df.columns)
    dtypes = [simplify_dtype(dtype) for dtype in df.dtypes]
    
    # Get total rows
    total_rows = len(df)
    
    if max_rows is None:
        max_rows = total_rows
    
    # Handle different sampling types
    if sampling_type == 'sequential':
        rows = df.iloc[:max_rows].values.tolist()
        
    elif sampling_type == 'random':
        if max_rows > total_rows:
            max_rows = total_rows
        random_indices = random.sample(range(total_rows), max_rows)
        rows = df.iloc[random_indices].values.tolist()
        
    elif sampling_type == 'segment':
        # Calculate number of full segments
        num_segments = total_rows // max_rows
        if segment > num_segments:
            raise ValueError(f"Segment {segment} exceeds maximum segments {num_segments}")
            
        start_idx = segment * max_rows
        end_idx = start_idx + max_rows
        rows = df.iloc[start_idx:end_idx].values.tolist()
    
    else:
        raise ValueError("sampling_type must be 'sequential', 'random' or 'segment'")
        
    # Build text representation
    text = "Dataset Description:\n"
    text += f"Total rows in dataset: {total_rows}\n"
    text += f"Rows included in this sample: {len(rows)}\n"
    text += f"Sampling method: {sampling_type}\n"
    if sampling_type == 'segment':
        text += f"Segment number: {segment}\n"
    
    text += "\nColumn Information:\n"
    for header, dtype in zip(headers, dtypes):
        # Get sample values for the column
        sample_vals = df[header].dropna().unique()[:3].tolist()
        sample_str = ", ".join(str(x) for x in sample_vals)
        
        text += f"- {header} ({dtype}): Example values: {sample_str}\n"
    
    text += "\nData Format:\n"
    text += "Each row contains values separated by commas in the following order:\n"
    text += ", ".join(headers) + "\n"
    
    text += "\nData Rows:\n"
    # Add rows with row numbers
    for i, row in enumerate(rows):
        text += f"Row {i+1}: " + ",".join(str(x) for x in row) + "\n"
        
    return text

# Example usage:
"""
# Sequential first 5 rows
text = csv_to_text('data.csv', max_rows=5, sampling_type='sequential')

# Random 5 rows
text = csv_to_text('data.csv', max_rows=5, sampling_type='random') 

# Get second segment of 5 rows
text = csv_to_text('data.csv', max_rows=5, sampling_type='segment', segment=1)
"""

# def csv_to_text_prompt(file_path, max_rows=10, mode='sequential', segment_number=0):
#     # Load the CSV into a DataFrame
#     df = pd.read_csv(file_path)
    
#     if mode == 'sequential':
#         # Extract sequential rows
#         rows = df.head(max_rows)
        
#     elif mode == 'random':
#         # Randomly sample rows
#         rows = df.sample(n=min(max_rows, len(df)))
        
#     elif mode == 'segmentation':
#         # Calculate the start and end indices for the segment
#         start_idx = segment_number * max_rows
#         end_idx = start_idx + max_rows
#         rows = df.iloc[start_idx:end_idx]
        
#     else:
#         raise ValueError("Mode must be 'sequential', 'random', or 'segmentation'.")
    
#     # Convert rows to a text representation
#     text_representation = ""
#     header = ", ".join(df.columns)
#     text_representation += f"Headers: {header}\n"
#     for index, row in rows.iterrows():
#         row_values = ", ".join(str(value) for value in row)
#         text_representation += f"Row {index + 1}: {row_values}\n"
    
#     return text_representation

# Example usage:
# text_prompt = csv_to_text_prompt('data.csv', max_rows=5, mode='random')
# print(text_prompt)







def main():
    parser = argparse.ArgumentParser(description="Convert CSV to text prompt.")
    parser.add_argument('file_path', type=str, help='Path to the CSV file.')
    parser.add_argument('--max_rows', type=int, default=10, help='Maximum number of rows to extract.')
    parser.add_argument('--mode', type=str, choices=['sequential', 'random', 'segment'], default='sequential', help='Mode of extraction.')
    parser.add_argument('--segment_number', type=int, default=0, help='Segment number for segmentation mode.')

    args = parser.parse_args()

    text_prompt = csv_to_text_prompt(
        csv_path=args.file_path,
        max_rows=args.max_rows,
        sampling_type=args.mode,
        segment=args.segment_number
    )

    print(text_prompt)

if __name__ == "__main__":
    main()