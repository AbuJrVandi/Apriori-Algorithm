import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules



# ----------------- Data Loading & Cleaning ----------------- #

# Load data
file_path = r"C:\Users\user\Desktop\MBA\Online Retail.xlsx"
mydata = pd.read_excel(file_path)

# Optimize data types
mydata['Quantity'] = mydata['Quantity'].astype(np.int32)
mydata['UnitPrice'] = mydata['UnitPrice'].astype(np.float32)

# ----------------- Data Cleaning ----------------- #

# Remove duplicates
mydata.drop_duplicates(inplace=True)

# Handle missing values
mydata.dropna(subset=['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice'], inplace=True)

# Formatting data
mydata['InvoiceDate'] = pd.to_datetime(mydata['InvoiceDate'])
mydata['TotalPrice'] = mydata['Quantity'] * mydata['UnitPrice']

# Convert to Market Basket Format
# Process in chunks if necessary
chunk_size = 10000  # Adjust based on your memory capacity
baskets = []

for start in range(0, len(mydata), chunk_size):
    end = start + chunk_size
    chunk = mydata.iloc[start:end]
    basket_chunk = (chunk
                    .groupby(['InvoiceNo', 'Description'])['Quantity']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('InvoiceNo'))
    baskets.append(basket_chunk)

# Combine all chunks into a single DataFrame
basket = pd.concat(baskets).groupby(level=0).sum()

# Convert values to 1s and 0s
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# ----------------- Transaction Data Transformation ----------------- #

# Function to transform data for a given country
def model_country(country, data, min_support=0.07, min_threshold=1):
    print(f"\n--- Processing data for {country} ---")
    
    # Filter and transform data for the given country
    basket = (data[data['Country'] == country]
              .groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
    
    # Convert quantities to binary (1 = item purchased, 0 = not purchased)
    basket_sets = basket.apply(lambda x: (x > 0).astype(int))
    basket_sets.drop(columns=['POSTAGE'], inplace=True, errors='ignore')
    #print(f"\nTransformed Basket (first 5 rows) for {country}:")
    #print(basket_sets.head())

# Call the model_country function with the cleaned data
model_country('United Kingdom', mydata)  # Replace 'United Kingdom' with the desired country

# Preprocess the data: Group by InvoiceNo and create a transaction list
basket = mydata.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')

# Convert quantities to binary values (1 for purchased, 0 for not purchased)
basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

# Generate frequent itemsets
frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Display the rules
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])