import os
import pandas as pd
import re
import networkx as nx
from collections import defaultdict
from multiprocessing import Pool, cpu_count

# Function to extract companies from '优化的专利权人' field
def extract_companies(owner_data):
    companies = owner_data.split(' | ')
    return companies

# Function to process a single Excel file and save the result as a CSV
def process_file(file_info):
    file_path, output_dir = file_info
    output_filename = os.path.join(output_dir, os.path.basename(file_path).replace('.xlsx', '_company_network.csv'))
    
    # Check if the result already exists, if yes, skip processing
    if os.path.exists(output_filename):
        print(f"Skipping {file_path}, result already exists.")
        return output_filename
    
    collaboration_by_year = defaultdict(nx.Graph)
    
    # Load the Excel file, skipping the first row
    df = pd.read_excel(file_path, skiprows=1)
    
    # Filter necessary columns
    df_filtered = df[['优化的专利权人', '申请日期']].dropna()
    
    # Extract companies
    df_filtered['companies'] = df_filtered['优化的专利权人'].apply(extract_companies)
    
    # Filter rows where company list is not empty
    df_filtered = df_filtered[df_filtered['companies'].apply(lambda x: len(x) > 0)]
    
    # Convert '申请日期' to datetime and extract the year, coercing errors
    df_filtered['year'] = pd.to_datetime(df_filtered['申请日期'], errors='coerce').dt.year
    
    # Filter out rows where 'year' is NaN or not an integer
    df_filtered = df_filtered.dropna(subset=['year'])
    df_filtered['year'] = df_filtered['year'].astype(int)  # Convert year to integer
    
    # Populate collaboration networks by year
    for index, row in df_filtered.iterrows():
        companies = row['companies']
        year = row['year']
        unique_companies = list(set(companies))
        
        for i in range(len(unique_companies)):
            for j in range(i + 1, len(unique_companies)):
                company_pair = tuple(sorted([unique_companies[i], unique_companies[j]]))  # Ensure A-B and B-A are the same
                for y in range(year, 2024):  # Assuming analysis up to 2024
                    if collaboration_by_year[y].has_edge(company_pair[0], company_pair[1]):
                        collaboration_by_year[y][company_pair[0]][company_pair[1]]['weight'] += 1
                    else:
                        collaboration_by_year[y].add_edge(company_pair[0], company_pair[1], weight=1)
    
    # Convert the collaboration networks into a DataFrame
    network_data = []
    for year, graph in collaboration_by_year.items():
        for edge in graph.edges(data=True):
            company1, company2, data = edge
            weight = data['weight']
            network_data.append([year, company1, company2, weight])
    
    # Save the result as a CSV
    df_network = pd.DataFrame(network_data, columns=['Year', 'Company1', 'Company2', 'Weight'])
    df_network.to_csv(output_filename, index=False)
    
    print(f"Processed {file_path} and saved to {output_filename}")
    
    return output_filename


# Function to process all files in the directory without multiprocessing
def process_all_files(directory, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the list of Excel files in the directory
    files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.xlsx')]
    
    # Prepare file info with output directory
    file_info = [(file, output_dir) for file in files]
    
    # Initialize a list to store the results
    result_files = []
    
    # Process each file sequentially
    for info in file_info:
        result_file = process_file(info)
        result_files.append(result_file)
    
    return result_files


# Function to merge all CSV files into one
def merge_csv_files(result_files, final_output_file):
    all_data = []
    
    for file in result_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            all_data.append(df)
    
    # Concatenate all DataFrames
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicate collaboration pairs (e.g., A-B and B-A should be counted as one)
    final_df['CompanyPair'] = final_df.apply(lambda x: ' | '.join(sorted([x['Company1'], x['Company2']])), axis=1)
    final_df = final_df.groupby(['Year', 'CompanyPair']).agg({'Weight': 'sum'}).reset_index()
    final_df[['Company1', 'Company2']] = final_df['CompanyPair'].str.split(' | ', expand=True)
    final_df = final_df.drop(columns=['CompanyPair'])
    
    # Save the final merged DataFrame
    final_df.to_csv(final_output_file, index=False)

# Main execution
if __name__ == '__main__':
    # Directory containing the input Excel files
    directory = './Dataset/NewData'
    
    # Directory to store the intermediate CSV results
    output_dir = './Dataset/CompanySubResults'  # Updated output directory for company sub-results
    
    # Output file for the final merged result
    final_output_file = './Dataset/company_collaboration_network.csv'  # Updated final output filename
    
    # Process all files and save intermediate CSVs
    result_files = process_all_files(directory, output_dir)
    
    # Merge all the CSV files into one final file
    merge_csv_files(result_files, final_output_file)
    
    print(f"Company collaboration network has been saved to {final_output_file}")
    print(f"Sub-result CSV files have been saved to {output_dir}")
