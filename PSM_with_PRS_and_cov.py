import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# --- Load data ---

icd10_codes = ('E10', 'E11', 'G30', 'J45', 'K90', 'K50', 'I10' , 'C43' , 'G35' , 'M81' , 'G20' , 'F20')
progect_path = "/sci/archive/michall/hadasak/Hakcathon_FH/data/PRS_data/"
covariates_path = progect_path+"data_participant_cov_for_PRS_PSM_numbers_only.csv"  # Update paths accordingly
prs_path = progect_path+"data_participant_PRS.csv"
phenotypes_path = progect_path + '_'.join(map(str, icd10_codes)) + "_phenotype_file.csv"
#output_path = progect_path+'_'.join(map(str, icd10_codes)) + "association_matrix.csv"



covariates = pd.read_csv(covariates_path)
prs_scores = pd.read_csv(prs_path)
phenotypes = pd.read_csv(phenotypes_path,sep="\t")

icd_mapping = {
    'alzheimer': 'G30',
    'asthma': 'J45',
    'coeliac': 'K90',
    'crohn': 'K50',
    'hypertension': 'I10',
    'melanoma': 'C43',
    'multiple_sclerosis': 'G35',
    'ostoporosis': 'M81',
    'parkinson': 'G20',
    'schizophrenia': 'F20',
    'type_1_diabetes': 'E10',
    'type_2_diabetes': 'E11'  # assumed typo in column name
}

# Keep other columns like 'Participant ID' unchanged
new_columns = {
    col: f"{icd_mapping[col]}_PRS" if col in icd_mapping else col
    for col in prs_scores.columns
}
prs_scores.rename(columns=new_columns, inplace=True)

print(covariates.head())
print(prs_scores.head())
print(phenotypes.head())

# --- Merge by ID ---
merged = covariates.merge(prs_scores, on="Participant ID").merge(phenotypes, left_on="Participant ID",right_on="FID")
print(merged.head())
print(merged.shape)
print(merged.shape)
print(merged.shape)
print(merged.shape)

# --- Get list of columns ---
covariate_cols = [col for col in covariates.columns if ((col != "Participant ID") &  (col != "Ethnic_group_classes"))]
prs_cols = [col for col in prs_scores.columns if col != "Participant ID"]
phenotype_cols = [col for col in phenotypes.columns if ((col != "FID") & (col != "sex") & (col != "year_of_birth"))]
all_columns = covariate_cols + prs_cols + phenotype_cols




# --- Helper function to detect binary vs continuous ---
def is_binary(series):
    unique_vals = series.dropna().unique()
    return len(unique_vals) == 2


def is_continuous(series):
    return np.issubdtype(series.dtype, np.number)


# --- Helper function for propensity score calculation ---
def calculate_propensity_scores(df, phenotype_col, covariate_cols):
    """
    Calculate propensity scores for a given phenotype based on the covariates.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    phenotype_col (str): The binary phenotype column.
    covariate_cols (list): List of covariate column names.

    Returns:
    pd.Series: Propensity scores for each participant.
    """
    # Extract the treatment variable (binary phenotype) and covariates
    X = df[covariate_cols]  # Covariates
    y = df[phenotype_col]  # Binary phenotype (0 or 1)

    # Drop rows where either the phenotype or any covariate is missing
    df_clean = df.dropna(subset=[phenotype_col] + covariate_cols)
    X_clean = df_clean[covariate_cols]
    y_clean = df_clean[phenotype_col]

    # Standardize the covariates (optional but helps with convergence)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Fit a logistic regression model to estimate the propensity scores
    model = LogisticRegression()
    model.fit(X_scaled, y_clean)

    # Return the propensity scores (predicted probabilities)
    propensity_scores = model.predict_proba(X_scaled)[:, 1]  # Get the probability for class '1'

    return propensity_scores


def convert_binary_columns_to_numeric(df):
    """
    Convert all binary categorical columns to 0 and 1 in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with binary categorical columns converted to 0 and 1.
    """
    for col in df.columns:
        # Check if the column is binary (i.e., has exactly two unique values)
        if df[col].nunique() == 2 and (df[col].dtype == 'object') :
            # Get unique values in the column
            unique_vals = df[col].dropna().unique()

            # Map unique values to 0 and 1
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}

            # Apply the mapping to the column
            df[col] = df[col].map(mapping)

    return df


# Example usage:
# Assuming `merged` is your DataFrame
merged = convert_binary_columns_to_numeric(merged)

merged_file = progect_path + "merged_file.csv"

merged.to_csv(merged_file, index=False)

merged = merged.dropna()

merged = merged[(merged['Ethnic_group_classes'] == 1) & (merged['genetic_sex'] >= 0)].drop(columns=['Ethnic_group_classes'])


# Check the result
print(merged.head())

# --- Loop over phenotypes and calculate propensity scores ---
for phenotype in phenotype_cols:
    # Check if phenotype is binary
    phenotype_data = merged[phenotype]
    if len(phenotype_data.unique()) == 2:  # Only for binary phenotypes
        print(f"Calculating propensity scores for {phenotype}...")


        print(covariate_cols)
        covariate_cols = [col for col in covariates.columns if ((col != "Participant ID") &  (col != "Ethnic_group_classes"))]
        print(covariate_cols)

        # Calculate the propensity scores using the covariates
        propensity_scores = calculate_propensity_scores(merged, phenotype, covariate_cols)

        covariate_cols_with_PRS = covariate_cols
        print(covariate_cols_with_PRS)

        covariate_cols_with_PRS.append(f"{phenotype}_PRS")


        print(covariate_cols_with_PRS)

        propensity_score_with_PRS = calculate_propensity_scores(merged, phenotype, covariate_cols_with_PRS)

        # Add the propensity scores to the dataframe
        merged[f"{phenotype}_propensity_score"] = propensity_scores
        merged[f"{phenotype}_propensity_score_with_PRS"] = propensity_score_with_PRS



# --- Save the dataframe with propensity scores ---
propensity_scores_path = progect_path + '_'.join(map(str, icd10_codes)) +"merged_with_propensity_scores_eroup_only_both_PRS_and_cov.csv"
merged.to_csv(propensity_scores_path, index=False)
print(f"Data with propensity scores saved to: {propensity_scores_path}")