import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import geopandas as gpd
import matplotlib.pyplot as plt

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


#####################################################################################
#                                     Checker
#####################################################################################


# Function to check if DataFrame has missing values
def check_no_missing_values(dataframe,name="Your Data"):
    if dataframe.isnull().sum().sum() == 0:
        print(f"{name} dont have missing values.")
    else:
        print(f"{name} contains missing values.")