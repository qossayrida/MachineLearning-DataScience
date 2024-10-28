import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import time


# 1. Document Missing Values
def document_missing_values(dataframe):
    missing_values = dataframe.isnull().sum()
    missing_percentage = (missing_values / len(dataframe)) * 100
    missing_report = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
    print("Missing Values Report:\n", missing_report[missing_report['Missing Values'] > 0])


# 2. Apply Missing Value Strategies
def handle_missing_values(dataframe):
    # Splitting the dataframe into numerical and categorical
    numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = dataframe.select_dtypes(include=['object']).columns

    # Strategy 1: Mean for numerical and most frequent for categorical
    imputer_mean_num = SimpleImputer(strategy='mean')
    imputer_most_frequent_cat = SimpleImputer(strategy='most_frequent')

    df_mean_num = pd.DataFrame(imputer_mean_num.fit_transform(dataframe[numerical_columns]), columns=numerical_columns)
    df_most_frequent_cat = pd.DataFrame(imputer_most_frequent_cat.fit_transform(dataframe[categorical_columns]),
                                        columns=categorical_columns)
    df_mean_most_frequent = pd.concat([df_mean_num, df_most_frequent_cat], axis=1)
    print(f"\nShape for data after mean imputation (numerical) and most frequent imputation (categorical): {df_mean_most_frequent.shape}")

    # Strategy 2: Median for numerical and forward/backward fill for categorical
    imputer_median_num = SimpleImputer(strategy='median')

    df_median_num = pd.DataFrame(imputer_median_num.fit_transform(dataframe[numerical_columns]),
                                 columns=numerical_columns)
    df_forward_fill_cat = dataframe[categorical_columns].ffill()
    df_backward_fill_cat = dataframe[categorical_columns].bfill()
    df_median_forward_fill = pd.concat([df_median_num, df_forward_fill_cat], axis=1)
    df_median_backward_fill = pd.concat([df_median_num, df_backward_fill_cat], axis=1)
    print(f"Shape for data after median imputation (numerical) and forward fill (categorical): {df_median_forward_fill.shape}")
    print(f"Shape for data after median imputation (numerical) and backward fill (categorical): {df_median_backward_fill.shape}")


    # Strategy 3: Dropping rows with missing values
    df_dropped = dataframe.dropna()
    print(f"Shape for data after dropping rows with missing values: {df_dropped.shape}\n\n")

    return df_mean_most_frequent, df_median_forward_fill, df_median_backward_fill, df_dropped


# 3. Feature Encoding
def encode_categorical_features(dataframe):
    # Ensure a deep copy of the dataframe to avoid SettingWithCopyWarning
    dataframe = dataframe.copy()

    # Combine 'Make' and 'Model' into one feature
    dataframe['Make_Model'] = dataframe['Make'] + "_" + dataframe['Model']

    # Combine 'County' and 'City' into one feature
    dataframe['County_City'] = dataframe['County'] + "_" + dataframe['City']

    # Drop the original columns 'Make', 'Model', 'County', and 'City'
    dataframe = dataframe.drop(columns=['Make', 'Model', 'County', 'City'])

    return dataframe



# 4. Normalize Numerical Features
def normalize_numerical_features(dataframe, method='min-max', feature_list=None):
    # If feature_list is provided, normalize only those columns, otherwise normalize all numerical columns
    if feature_list:
        numerical_columns = feature_list
    else:
        numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns

    # Choose the normalization method: Min-Max or Z-Score
    if method == 'min-max':
        scaler = MinMaxScaler()
    elif method == 'z-score':
        scaler = StandardScaler()
    else:
        print(f"Invalid method {method}. No normalization applied.")
        return dataframe

    # Apply the scaler to the specified columns
    dataframe[numerical_columns] = scaler.fit_transform(dataframe[numerical_columns])

    return dataframe



# 5. Calculate summary statistics (mean, median, standard deviation)
def calculate_summary_statistics(dataframe):
    # Select only numerical columns
    numerical_columns = dataframe.select_dtypes(include=['float64', 'int64'])

    # Calculate mean, median, and standard deviation for each numerical column
    summary_stats = pd.DataFrame({
        'Mean': numerical_columns.mean(),
        'Median': numerical_columns.median(),
        'Standard Deviation': numerical_columns.std()
    })

    # Print the summary statistics
    print("\nSummary Statistics for Numerical Features:")
    print(summary_stats)

    return summary_stats

# 6.
def visualize_summary_distribution_one (dataframe ,shapefile=r"tl_2023_53_cousub\tl_2023_53_cousub.shp"):

    # Step 1: Load the shapefile for Washington State counties
    # To get the map: https://catalog.data.gov/dataset/tiger-line-shapefile-current-state-washington-county-subdivision
    washington_counties = gpd.read_file(shapefile)

    # Step 2: Ensure the CRS is appropriate (EPSG:4326 is a common lat/lon projection)
    washington_counties = washington_counties.to_crs(epsg=4326)

    # Step 3: Count the number of electric vehicles per county from the dataset
    county_counts = dataframe['City'].value_counts().reset_index()
    county_counts.columns = ['City', 'vehicle_count']

    # Step 4: Merge the counts with the Washington counties shapefile based on county names
    # Assuming the 'NAME' column in the shapefile contains the county names
    washington_counties = washington_counties.merge(county_counts, left_on='NAME', right_on='City', how='left')

    # Step 5: Replace NaN values with 0 for counties that may not have electric vehicle data
    # Instead of using inplace=True, assign the result back to the column
    washington_counties['vehicle_count'] = washington_counties['vehicle_count'].fillna(-10000)

    # Step 6: Plot the map with the vehicle counts data
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    washington_counties.plot(column='vehicle_count', cmap='YlGnBu', linewidth=0.8, ax=ax, edgecolor='0.8')

    # Add title
    plt.title('Electric Vehicle Counts by Washington Counties', fontsize=15)

    # Show the plot
    plt.show()


# 6.
def visualize_summary_distribution_two(dataframe, shapefile=r"tl_2023_53_cousub\tl_2023_53_cousub.shp"):
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df_copy = dataframe.copy()

    # Extract the coordinates from the 'Vehicle Location' column
    df_copy['Longitude'] = df_copy['Vehicle Location'].apply(lambda x: float(x.split()[1].strip('()')))
    df_copy['Latitude'] = df_copy['Vehicle Location'].apply(lambda x: float(x.split()[2].strip('()')))

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(df_copy, geometry=gpd.points_from_xy(df_copy.Longitude, df_copy.Latitude))
    gdf.set_crs(epsg=4326, inplace=True)

    # Load a shapefile for Washington State for the base map
    washington_counties = gpd.read_file(shapefile)
    washington_counties = washington_counties.to_crs(epsg=4326)

    # Plot the map with vehicle points
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the counties of Washington state
    washington_counties.plot(ax=ax, color='lightgrey', edgecolor='black')

    # Plot the EV points
    gdf.plot(ax=ax, color='blue', markersize=5, label='EV Locations')

    # Add titles and labels
    plt.title("Spatial Distribution of Electric Vehicles in Washington State", fontsize=15)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()

    # Show the map
    plt.show()



# 7.Analyze the popularity of different EV models (categorical data) and identify any trends.
def analyze_ev_model_popularity(dataframe):
    # Check if 'Make' and 'Model' columns exist
    if 'Make' not in dataframe.columns or 'Model' not in dataframe.columns:
        print("Error: 'Make' or 'Model' column not found in the DataFrame.")
        print("Available columns:", dataframe.columns)
        return

    # Group by 'Make' and 'Model' and count occurrences
    model_popularity = dataframe.groupby(['Make', 'Model']).size().reset_index(name='Count')

    # Sort the models by popularity (count) in descending order
    model_popularity = model_popularity.sort_values(by='Count', ascending=False)

    # Get the top 15 models and combine the rest into 'Other Models'
    top_models = model_popularity.head(15)
    other_models_count = model_popularity['Count'].sum() - top_models['Count'].sum()

    # Create a new DataFrame for pie chart data
    other_models_df = pd.DataFrame({'Make': ['Other Models'], 'Model': [''], 'Count': [other_models_count]})
    pie_chart_data = pd.concat([top_models, other_models_df], ignore_index=True)

    # Print the top 5 models with aligned formatting (with header)
    print("\n\nTop 5 Electric Vehicle Models by Popularity:")
    print(f"{'Make':<15} {'Model':<15} {'Count':<7}")
    for index, row in top_models.head(5).iterrows():
        print(f"{row['Make']:<15} {row['Model']:<15} {row['Count']:<7}")

    # Create a pie chart for top models and 'Other Models'
    plt.figure(figsize=(12, 10))
    plt.pie(pie_chart_data['Count'],
            labels=pie_chart_data['Make'] + (": " + pie_chart_data['Model'].replace('', ' ')),
            autopct='%1.1f%%',
            startangle=140,
            colors=plt.cm.tab20.colors)

    # Set the title and display the pie chart
    plt.title('Electric Vehicle Models by Popularity\n')
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
    plt.show()


# 8. Investigate the relationship between every pair of numeric features.

def investigate_correlations(df, method='pearson', threshold=0.8):
    # Extract numeric features from the DataFrame
    numeric_df = df.select_dtypes(include='number')

    # Calculate correlation matrix
    correlation_matrix = numeric_df.corr(method=method)

    # Visualize the correlation matrix using a heatmap
    plt.figure(figsize=(14, 13))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f'Correlation Matrix ({method.capitalize()} method)', fontsize=16)
    plt.show()

    # Explain the results
    print("\n### Correlation Analysis ###\n")
    corr_matrix_no_self = correlation_matrix.copy()

    # Remove self-correlations
    np.fill_diagonal(corr_matrix_no_self.values, np.nan)

    strong_correlations = corr_matrix_no_self[
        (corr_matrix_no_self > threshold) | (corr_matrix_no_self < -threshold)
    ]

    if strong_correlations.isnull().all().all():
        print(
            f"No strong correlations (greater than {threshold} or less than -{threshold}) found between numeric features."
        )
    else:
        print(f"Significant Correlations (greater than {threshold} or less than -{threshold}):\n")
        for i, row in strong_correlations.iterrows():
            for j, value in row.items():
                if not pd.isna(value):
                    print(f" - {i} and {j} have a correlation of {value:.2f}")

    print("\nCorrelations closer to 1 indicate a strong positive relationship, while correlations closer to -1 indicate a strong negative relationship. Correlations close to 0 suggest no linear relationship.")

    return correlation_matrix


# 9.
"""
    Create data exploration visualizations, including histograms for specified numerical features,
    scatter plots for selected feature pairs, and boxplots to understand distributions by a specified categorical feature.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame containing the data to visualize.
    - numerical_features (list): List of numerical column names to visualize. If None, uses all numerical columns.
"""
def explore_data_visualizations(dataframe, numerical_features=None):


    # Select numerical columns
    if numerical_features is None:
        numerical_features = dataframe.select_dtypes(include=['float64', 'int64']).columns
    else:
        numerical_features = [col for col in numerical_features if col in dataframe.columns]

    # 1. Histograms for numerical features
    def plot_histograms(dataframe):
        for col in numerical_features:
            plt.figure(figsize=(8, 6))
            plt.hist(dataframe[col], bins=20, color='skyblue', edgecolor='black')
            plt.title(f"Distribution of {col}", fontsize=16, weight='bold')
            plt.xlabel(col, fontsize=14)
            plt.ylabel("Frequency", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
            time.sleep(0.5)

    # 2. Scatter plots between selected pairs of numerical features
    def plot_scatter_matrix(dataframe):
        num_cols = len(numerical_features)

        # Create separate scatter plots for each pair of numerical columns
        for i in range(num_cols):
            for j in range(i + 1, num_cols):
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=dataframe, x=numerical_features[i], y=numerical_features[j], alpha=0.6, s=50, color="teal",
                                edgecolor='black')
                plt.title(f"Scatter Plot of {numerical_features[i]} vs {numerical_features[j]}", fontsize=16, weight='bold')
                plt.xlabel(numerical_features[i], fontsize=14)
                plt.ylabel(numerical_features[j], fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.show()
                time.sleep(0.5)

    # 3. Boxplot for each numerical feature by a chosen categorical feature
    def plot_boxplots(dataframe,categorical_feature):
        for num_col in numerical_features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=dataframe, x=categorical_feature, y=num_col, color="lightblue", fliersize=3)
            plt.title(f"Boxplot of {num_col} by {categorical_feature}", fontsize=16, weight='bold')
            plt.xlabel(categorical_feature, fontsize=14)
            plt.ylabel(num_col, fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
            time.sleep(0.5)

    # Execute the visualizations
    print("Generating histograms for specified numerical features...")
    plot_histograms(dataframe)

    print("Generating scatter plot matrix for specified numerical features...")
    plot_scatter_matrix(dataframe)

    if 'Make' in dataframe.columns:
        print("Generating boxplots for numerical features by 'Make' (categorical feature)...")
        plot_boxplots(dataframe, 'Make')
    else:
        print("Generating boxplots for numerical features by 'Make_Model' (categorical feature)...")
        plot_boxplots(dataframe, 'Make_Model')




#####################################################################################
#                                     Checker
#####################################################################################


# Function to check if DataFrame has missing values
def check_no_missing_values(dataframe,name="Your Data"):
    if dataframe.isnull().sum().sum() == 0:
        print(f"{name} dont have missing values.")
    else:
        print(f"{name} contains missing values.")