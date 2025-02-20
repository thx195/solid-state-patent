import os
import pandas as pd

# Function to sort company1 and company2 columns alphabetically
def sort_companies(df):
    df[['Company1', 'Company2']] = df.apply(
        lambda row: sorted([row['Company1'], row['Company2']]), axis=1, result_type="expand"
    )
    return df

# Function to merge all CSV files into one
def merge_csv_files(directory, final_output_file):
    all_data = []
    
    # Traverse the directory to find all CSV files
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty:  # Check if the DataFrame is not empty
                        all_data.append(df)
                    else:
                        print(f"Warning: {file_path} is empty and has been skipped.")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    if len(all_data) == 0:
        print("Error: No valid data found to concatenate.")
        return
    
    # Concatenate all DataFrames
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Sort company1 and company2 in each row to avoid duplicate pairs in reverse order
    final_df = sort_companies(final_df)
    
    # Group by 'Year', 'Company1', and 'Company2', then sum the 'Weight'
    final_df = final_df.groupby(['Year', 'Company1', 'Company2'], as_index=False).agg({'Weight': 'sum'})
    
    # Save the final merged DataFrame to a CSV file
    final_df.to_csv(final_output_file, index=False)
    print(f"Final merged file has been saved to {final_output_file}")

# Main execution
if __name__ == '__main__':
    # Directory containing the sub-result CSV files
    directory = './Dataset/CompanySubResults'
    
    # Output file for the final merged result
    final_output_file = './Dataset/company_collaboration_network_final.csv'
    
    # Merge all the CSV files into one final file
    merge_csv_files(directory, final_output_file)
