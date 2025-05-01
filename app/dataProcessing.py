import pandas as pd
import numpy as np
import pickle
import json
import os

def process_data(prepared_data):

    one_dp_dict = prepared_data.copy()

    # Converting dictionary to dataframe
    X_one = pd.DataFrame([one_dp_dict])

    X_one_processed = X_one.copy()

    X_one_processed['minority_ratio'] = X_one_processed['minority_population'] / X_one_processed['population']
    X_one_processed['ownership_rate'] = X_one_processed['number_of_owner_occupied_units'] / X_one_processed['number_of_1_to_4_family_units']
    X_one_processed['tract_income_deviation'] = X_one_processed['tract_to_msamd_income'] - 1
    X_one_processed['wealth_density'] = X_one_processed['ownership_rate'] * X_one_processed['tract_to_msamd_income']

    bin_labels = ['Income_Level_1', 'Income_Level_2']
    bin_edges = [45600., 53200., 61400.]

    X_one_processed['income_tier'] = pd.cut(X_one_processed['hud_median_family_income'], bins=bin_edges, labels=bin_labels, include_lowest=True)

    # drop columns post feature engineering

    cols_to_drop2 = ['census_tract_number', 'population',
                    'minority_population', 'hud_median_family_income',
                    'tract_to_msamd_income', 'number_of_owner_occupied_units',
                    'number_of_1_to_4_family_units']

    X_one_processed = X_one_processed.drop(columns=cols_to_drop2, errors='ignore') # 21 cols

    # Log Transformation

    X_one_processed["applicant_income_000s"] = pd.to_numeric(X_one_processed["applicant_income_000s"], errors="coerce")

    X_one_processed["loan_amount_000s"] = pd.to_numeric(X_one_processed["loan_amount_000s"], errors="coerce")

    X_one_processed["applicant_income_000s"] = X_one_processed["applicant_income_000s"].apply(np.log)
    X_one_processed["loan_amount_000s"] = X_one_processed["loan_amount_000s"].apply(np.log)

    numerical_cols = ['loan_amount_000s', 'applicant_income_000s', 'minority_ratio',
                  'ownership_rate', 'tract_income_deviation', 'wealth_density']
    categorical_cols = [
        'agency_name', 'property_type_name', 'owner_occupancy_name',
        'preapproval_name', 'county_name',
        'applicant_ethnicity_name', 'co_applicant_ethnicity_name',
        'applicant_race_name_1', 'co_applicant_race_name_1',
        'applicant_sex_name', 'co_applicant_sex_name',
        'purchaser_type_name', 'hoepa_status_name', 'lien_status_name',
        'income_tier'
    ]

    ##### One hot encoding

    # load ohe
    
    with open(os.path.join(os.getcwd(), "app", "one_hot_encoder.pkl"), "rb") as file:
        ohe = pickle.load(file)

    # Load cat_feature_names
    
    with open(os.path.join(os.getcwd(), "app", "one_hot_encoder_cat_feature_names.pkl"), "rb") as file:
        cat_feature_names = pickle.load(file)

    X_one_cat_encoded = ohe.transform(X_one_processed[categorical_cols])

    X_one_cat_encoded_df = pd.DataFrame(X_one_cat_encoded, columns=cat_feature_names, index=X_one_processed.index) # 144 cols

    ######## Standardization

    # Load scaler from pickle
    
    with open(os.path.join(os.getcwd(), "app", "standard_scalar_obj.pkl"), "rb") as file:
        scaler = pickle.load(file)

    X_one_numeric_scaled = scaler.transform(X_one_processed[numerical_cols])
    X_one_numeric_scaled_df = pd.DataFrame(X_one_numeric_scaled, columns=numerical_cols, index=X_one_processed.index)

    # concatenate the dataframes
    X_one_final = pd.concat([X_one_numeric_scaled_df, X_one_cat_encoded_df], axis=1)

    ########## PCA
    # Load PCA 

    
    with open(os.path.join(os.getcwd(), "app", "pca_obj.pkl"), "rb") as file:
        pca = pickle.load(file)

    X_one_pca = pca.transform(X_one_final) 

    return X_one_pca