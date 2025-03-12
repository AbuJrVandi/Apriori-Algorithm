import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules


# ----------------- Data Loading & Cleaning ----------------- #

# Load data
file_path = r"/Users/user/Desktop/Dr. A. J/MBA/Online Retail.xlsx"
mydata = pd.read_excel(file_path)

# Define countries
countries = ["Germany", "France", "Netherlands"]


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
    
    # ----------------- Model Training (Apriori Algorithm) ----------------- #
    
    # Generate frequent itemsets
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold, num_itemsets=2)
    
    print(f"\nFrequent Itemsets (first 5 rows) for {country}:")
    print(frequent_itemsets.head())
    
    print(f"\nAssociation Rules (first 5 rows) for {country}:")
    print(rules.head())
    
    return basket_sets, frequent_itemsets, rules

# Example usage for Germany, France, and Netherlands
basket_sets_germany, freq_germany, rules_germany = model_country("Germany", mydata)
basket_sets_france, freq_france, rules_france = model_country("France", mydata)
basket_sets_netherlands, freq_netherlands, rules_netherlands = model_country("Netherlands", mydata)

# ----------------- Data Visualization ----------------- #

# Ensure basket_data is not empty and contains valid data
for country in countries:
    plt.figure(figsize=(10, 6))
    sns.heatmap(freq_germany.pivot_table(index='itemsets', values='support'), cmap="Blues", annot=True)
    plt.title(f"Heatmap of Frequent Itemsets - {country}")
    plt.xticks(rotation=90)
    plt.show()

# Bar chart for top 10 items by frequency
for country in countries:
    top_items = freq_germany.sort_values(by="support", ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_items["support"], y=top_items["itemsets"].astype(str), palette="viridis")
    plt.title(f"Top 10 Frequent Items - {country}")
    plt.xlabel("Support")
    plt.ylabel("Itemsets")
    plt.show()

# Bar chart for top 10 items by frequency
top_items = basket_sets_germany.sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_items.index, y=top_items.values, palette="coolwarm")
plt.xticks(rotation=90)
plt.xlabel("Item Description")
plt.ylabel("Frequency")
plt.title("Top 10 Most Frequent Items")
plt.show()

# ----------------- Product Recommendation System ----------------- #

def recommend_products(product, rules_df, num_recommendations=5):
    """
    Function to recommend products based on association rules.
    
    :param product: Product name (must match Description in dataset)
    :param rules_df: Association rules DataFrame
    :param num_recommendations: Number of recommendations to return
    :return: List of recommended products
    """
    # Filter rules where the product appears in the antecedents
    recommendations = rules_df[rules_df['antecedents'].apply(lambda x: product in x)]
    
    # Sort by lift (higher lift means stronger association)
    recommendations = recommendations.sort_values(by='lift', ascending=False)
    
    if recommendations.empty:
        print(f"\nNo strong association rules found for: {product}")
        return []
    
    # Extract recommended products
    recommended_products = set()
    for consequents in recommendations['consequents']:
        recommended_products.update(consequents)
    
    recommended_products.discard(product)  # Remove input product from recommendations
    
    # Limit recommendations to the requested number
    return list(recommended_products)[:num_recommendations]

# Example usage: Get recommendations for a specific product in Germany
product_to_check = "ROUND SNACK BOXES SET OF4 WOODLAND"
recommended_items = recommend_products(product_to_check, rules_germany)

# Print recommendations
print(f"\nRecommended Products for '{product_to_check}': {recommended_items}")
