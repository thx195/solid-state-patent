import pandas as pd

# Load the existing result CSV file
input_file = './Dataset/country_collaboration_network.csv'
df = pd.read_csv(input_file)

# Fill NaN values with empty strings or drop them
df = df.dropna(subset=['Country1', 'Country2'])

# Create a new DataFrame for merged results
merged_data = []

# Create normalized country pairs, ensuring all values are strings
df['CountryPair'] = df.apply(lambda row: tuple(sorted([str(row['Country1']), str(row['Country2'])])), axis=1)
grouped = df.groupby(['Year', 'CountryPair'], as_index=False).agg({'Weight': 'sum'})

# Split the country pairs back into two columns
grouped[['Country1', 'Country2']] = pd.DataFrame(grouped['CountryPair'].tolist(), index=grouped.index)

# Drop the CountryPair column
grouped = grouped.drop(columns=['CountryPair'])

# Save the merged results to a new CSV file
output_file = './Dataset/merged_country_collaboration_network.csv'
grouped.to_csv(output_file, index=False)

print(f"Merged collaboration network has been saved to {output_file}")
