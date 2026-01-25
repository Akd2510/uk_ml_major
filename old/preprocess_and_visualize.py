import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD THE DATASET
# Ensure 'Global_Landslide_Catalog_Export_rows.csv' is in the same folder as this script
file_name = 'Global_Landslide_Catalog_Export_rows.csv'
df = pd.read_csv(file_name)

# 2. FILTER FOR UTTARAKHAND
# We filter by country first, then by the specific state division
uk_data = df[(df['country_name'] == 'India') & 
             (df['admin_division_name'] == 'Uttarakhand')].copy()

# 3. DATA PRE-PROCESSING
# Convert the 'event_date' column to a proper datetime format
uk_data['event_date'] = pd.to_datetime(uk_data['event_date'])
uk_data['year'] = uk_data['event_date'].dt.year
uk_data['month'] = uk_data['event_date'].dt.month

# 4. SAVE THE FILTERED DATA
# This creates a smaller, more manageable CSV for your ML model
output_file = 'Uttarakhand_Landslide_Data.csv'
uk_data.to_csv(output_file, index=False)
print(f"Successfully saved {len(uk_data)} records to {output_file}")

# 5. DATA ANALYSIS & VISUALIZATION
# Set the visual style
sns.set_theme(style="whitegrid")

# Plot 1: Landslide Occurrences by Month (Seasonality)
plt.figure(figsize=(10, 6))
month_counts = uk_data['month'].value_counts().sort_index()
sns.barplot(x=month_counts.index, y=month_counts.values, palette='flare')
plt.title('Monthly Distribution of Landslides in Uttarakhand (Monsoon Spikes)')
plt.xlabel('Month (1=Jan, 12=Dec)')
plt.ylabel('Number of Events')
plt.xticks(ticks=range(0, 12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.savefig('uk_monthly_trends.png')
print("Monthly trend plot saved as 'uk_monthly_trends.png'")

# Plot 2: Geographic Distribution
plt.figure(figsize=(10, 8))
sns.scatterplot(data=uk_data, x='longitude', y='latitude', hue='landslide_trigger', size='landslide_size', alpha=0.7)
plt.title('Geospatial Mapping of Recorded Landslides in Uttarakhand')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('uk_landslide_map.png')
print("Geospatial map saved as 'uk_landslide_map.png'")

print("\nTop 5 Landslide Triggers in Uttarakhand:")
print(uk_data['landslide_trigger'].value_counts().head())