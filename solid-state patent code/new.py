import os
import pandas as pd
import re
import networkx as nx
from collections import defaultdict

# Function to extract countries from '发明人 - 带有地址' field
def extract_countries(inventor_data):
    inventors = inventor_data.split(' | ')
    countries = []
    for inventor in inventors:
        parts = inventor.split('|')
        if len(parts) > 1:
            address_parts = parts[-1].split(', ')
            if address_parts and re.match(r'^[A-Z]{2}$', address_parts[-1]):
                countries.append(address_parts[-1])
    return countries

# Function to clean country list
def clean_countries(country_list):
    return [country for country in country_list if re.match(r'^[A-Z]{2}$', country)]

# Function to process a single Excel file and save the result as a CSV
def process_file(file_path, output_dir):
    output_filename = os.path.join(output_dir, os.path.basename(file_path).replace('.xlsx', '_network.csv'))
    
    # If the output file already exists, skip processing
    if os.path.exists(output_filename):
        print(f"Skipping {file_path}, already processed.")
        return output_filename

    collaboration_by_year = defaultdict(nx.Graph)
    
    # Load the Excel file, skipping the first row
    df = pd.read_excel(file_path, skiprows=1)
    
    # Filter necessary columns
    df_filtered = df[['发明人 - 带有地址', '申请日期']].dropna()
    
    # Extract countries
    df_filtered['countries'] = df_filtered['发明人 - 带有地址'].apply(extract_countries)
    
    # Clean the countries column
    df_filtered['countries'] = df_filtered['countries'].apply(clean_countries)
    
    # Filter rows where country list is not empty
    df_filtered = df_filtered[df_filtered['countries'].apply(lambda x: len(x) > 0)]
    
    # Convert '申请日期' to datetime and extract the year
    df_filtered['year'] = pd.to_datetime(df_filtered['申请日期']).dt.year
    
    # Populate collaboration networks by year
    for index, row in df_filtered.iterrows():
        countries = row['countries']
        year = row['year']
        unique_countries = list(set(countries))
        
        for i in range(len(unique_countries)):
            for j in range(i + 1, len(unique_countries)):
                for y in range(year, 2024):  # Assuming analysis up to 2024
                    if collaboration_by_year[y].has_edge(unique_countries[i], unique_countries[j]):
                        collaboration_by_year[y][unique_countries[i]][unique_countries[j]]['weight'] += 1
                    else:
                        collaboration_by_year[y].add_edge(unique_countries[i], unique_countries[j], weight=1)
    
    # Convert the collaboration networks into a DataFrame
    network_data = []
    for year, graph in collaboration_by_year.items():
        for edge in graph.edges(data=True):
            country1, country2, data = edge
            weight = data['weight']
            network_data.append([year, country1, country2, weight])
    
    # Save the result as a CSV
    df_network = pd.DataFrame(network_data, columns=['Year', 'Country1', 'Country2', 'Weight'])
    df_network.to_csv(output_filename, index=False)
    
    print(f"Processed {file_path} and saved to {output_filename}.")
    return output_filename

# Function to process all files in the directory
def process_all_files(directory, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the list of Excel files in the directory
    files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.xlsx')]
    
    result_files = []
    
    # Process each file one by one
    for file in files:
        result_file = process_file(file, output_dir)
        result_files.append(result_file)
    
    return result_files

# Function to merge all CSV files into one
def merge_csv_files(result_files, final_output_file):
    all_data = []
    
    for file in result_files:
        df = pd.read_csv(file)
        all_data.append(df)
    
    # Concatenate all DataFrames
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Save the final merged DataFrame
    final_df.to_csv(final_output_file, index=False)
    print(f"Final merged file saved to {final_output_file}")

# Main execution
if __name__ == '__main__':
    # Directory containing the input Excel files
    directory = './Dataset/NewData'
    
    # Directory to store the intermediate CSV results
    output_dir = './Dataset/SubResults'
    
    # Output file for the final merged result
    final_output_file = './Dataset/country_collaboration_network.csv'
    
    # Process all files and save intermediate CSVs
    result_files = process_all_files(directory, output_dir)
    
    # Merge all the CSV files into one final file
    merge_csv_files(result_files, final_output_file)
    
    print(f"Country collaboration network has been saved to {final_output_file}")
    print(f"Sub-result CSV files have been saved to {output_dir}")
