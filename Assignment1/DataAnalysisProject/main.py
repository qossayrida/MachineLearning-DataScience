import pandas as pd
import Collection

#############################################################################
#                                Load dataset
#############################################################################

df = pd.read_csv("Electric_Vehicle_Population_Data.csv")



#############################################################################
#                         Document Missing Values
#############################################################################

Collection.document_missing_values(df)


#############################################################################
#          Handle missing values by mean imputation and dropping rows
#############################################################################

df_mean_most_frequent, df_median_forward_fill, df_median_backward_fill, df_dropped = Collection.handle_missing_values(df)

Collection.check_no_missing_values(df_mean_most_frequent,"mean_most_frequent")
Collection.check_no_missing_values(df_median_forward_fill,"median_forward_fill")
Collection.check_no_missing_values(df_median_backward_fill,"median_backward_fill")
Collection.check_no_missing_values(df_mean_most_frequent,"mean_most_frequent")


#############################################################################
#                            Feature Encoding
#############################################################################

df_dropped_encode = Collection.encode_categorical_features(df_dropped)
df_mean_most_frequent_encode = Collection.encode_categorical_features(df_mean_most_frequent)

#############################################################################
#                             Normalization
#############################################################################

df_dropped_encode_normalize = Collection.normalize_numerical_features(df_dropped_encode)
df_normalized = Collection.normalize_numerical_features(df, method='z-score', feature_list=['Electric Range'])

#############################################################################
#                        Descriptive Statistics
#############################################################################

Collection.calculate_summary_statistics(df_dropped_encode)


#############################################################################
#                         Spatial Distribution
#############################################################################

Collection.visualize_summary_distribution_one(df_dropped)
Collection.visualize_summary_distribution_two(df_dropped)

#############################################################################
#                           Model Popularity
#############################################################################

Collection.analyze_ev_model_popularity(df_dropped)

#############################################################################
#    Investigate the relationship between every pair of numeric features
#############################################################################

Collection.investigate_correlations(df_dropped)


#############################################################################
#
#############################################################################
"""
Numerical features in our data:
['Postal Code', 'Model Year', 'Electric Range', 'Base MSRP',
'Legislative District', 'DOL Vehicle ID', '2020 Census Tract']

If you don't pass numerical_features list the code will plot for all numerical features 
This causes this error to appear:
HTTP Error 429: Too Many Requests, usually indicates that too many requests have been sent to the server 
To avoid this increase sleep time after each plot to 2 seconds
"""

Collection.explore_data_visualizations(df_dropped, numerical_features=['DOL Vehicle ID', 'Electric Range'])

#############################################################################
#
#############################################################################

Collection.visualize_ev_distribution_by_location(df_dropped)


#############################################################################
#
#############################################################################

Collection.temporal_analysis(df_dropped)

#############################################################################
#                            save data and exit
#############################################################################

df_dropped_encode_normalize.to_csv('Cleaned_Electric_Vehicle_Data.csv', index=False)
print("\n\nCleaned data saved.")


