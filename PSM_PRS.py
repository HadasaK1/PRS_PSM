import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
from statsmodels.stats.contingency_tables import Table2x2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import fisher_exact
import math
import argparse



#####
#file aims:
#1. Create the data for Disease1 and Diseas2 with all covariances.
#2. Calculate propensity score based on the classical covariance.
#3. Calculate propensity score based on classical + PRS.
#4. Plot density plot and calculate SMD for cases amd control in the outcome disease and in the control diseases.
#5. Do matching for cases and controls based on each PS (PRS, classical, and combined)
#6. Running Fisher's test on the matched data.
#7. Save the data and the results to a CSV file.


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


def calc_smd(series_cases, series_controls):
    """Compute Standardized Mean Difference (SMD)."""
    mean_cases = series_cases.mean()
    mean_controls = series_controls.mean()
    var_cases = series_cases.var()
    var_controls = series_controls.var()
    return (mean_cases - mean_controls) / np.sqrt((var_cases + var_controls) / 2)


def plot_multiple_ps(df, ps_cols, exposure_col, outcome_col, output_path):
    """
    Create a grid of density plots for multiple PS variables,
    for both exposure and outcome case/control.

    Parameters:
    -----------
    df : pd.DataFrame
        Data with PS columns and case/control labels
    ps_cols : list of str
        List of propensity score column names
    exposure_col : str
        Column name for exposure case/control (0/1)
    outcome_col : str
        Column name for outcome case/control (0/1)
    output_path : str
        File path to save the combined plot
    """
    n_ps = len(ps_cols)
    fig, axes = plt.subplots(nrows=2, ncols=n_ps, figsize=(6 * n_ps, 10), sharex=True, sharey=False)

    for i, ps in enumerate(ps_cols):
        # Scale PS column to 0-1
        ps_scaled = (df[ps] - df[ps].min()) / (df[ps].max() - df[ps].min())
        df[ps] = ps_scaled

    for i, ps in enumerate(ps_cols):

        # Exposure plot
        cases = df.loc[df[exposure_col] == 2, ps]
        controls = df.loc[df[exposure_col] == 1, ps]

        smd = calc_smd(cases, controls)

        sns.kdeplot(
            data=df, x=ps, hue=exposure_col, fill=True, common_norm=False, alpha=0.4, ax=axes[0, i]
        )
        axes[0, i].set_title(f"{ps} - Exposure\nSMD={smd:.3f}")
        axes[0, i].set_xlabel("Propensity Score")
        axes[0, i].set_ylabel("Density")
        axes[0, i].legend(title="Exposure", labels=["Case","Control"])

        # Outcome plot
        cases = df.loc[df[outcome_col] == 2, ps]
        controls = df.loc[df[outcome_col] == 1, ps]
        smd = calc_smd(cases, controls)
        print("Number of cases:", len(cases))
        print("Number of controls:", len(controls))
        print("Missing values in cases:", cases.isna().sum())
        print("Missing values in controls:", controls.isna().sum())


        sns.kdeplot(
            data=df, x=ps, hue=outcome_col, fill=True, common_norm=False, alpha=0.4, ax=axes[1, i]
        )
        axes[1, i].set_title(f"{ps} - Outcome\nSMD={smd:.3f}")
        axes[1, i].set_xlabel("Propensity Score")
        axes[1, i].set_ylabel("Density")
        axes[1, i].legend(title="Outcome", labels=["Case","Control"])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved 6-panel density plot to {output_path}")




# Matching function
def match_by_score(df, disease_col, score_col,output_path):


    treated = df[(df[disease_col] == 2) & (~df[score_col].isna())].copy()
    control = df[(df[disease_col] == 1) & (~df[score_col].isna())].copy()

    print(treated)
    print(control)


    treated_scores = treated[[score_col]].values
    control_scores = control[[score_col]].values

    used_control_indices = set()
    matched_pairs = []

    for i, t_score in enumerate(treated_scores):
        available_indices = [idx for idx in range(len(control_scores)) if idx not in used_control_indices]
        if not available_indices:
            break

        available_scores = control_scores[available_indices]
        distances = np.abs(available_scores - t_score)
        closest_idx_in_available = np.argmin(distances)
        #if closest_idx_in_available > 0.5 or closest_idx_in_available < -0.5:
        #    continue
        closest_control_idx = available_indices[closest_idx_in_available]
        used_control_indices.add(closest_control_idx)
        matched_pairs.append((treated.index[i], control.index[closest_control_idx]))

    # Build matched dataset
    matched_treated = treated.loc[[i for i, _ in matched_pairs]]
    matched_controls = control.loc[[j for _, j in matched_pairs]]
    matched_df = pd.concat([matched_treated, matched_controls], axis=0)

    # Counts
    #cases_phen1 = matched_df[matched_df[disease_col] == 2]["Participant ID"].nunique()
    #unique_controls_phen1 = matched_df[matched_df[disease_col] == 1]["Participant ID"].nunique()
    cases_phen1 = matched_df[matched_df[disease_col] == 2]
    unique_controls_phen1 = matched_df[matched_df[disease_col] == 1]

    matched_df.to_csv(output_path)


    return matched_df, cases_phen1, unique_controls_phen1

# Fisher test function
def run_fisher(matched_df, phen1, phen2,method):
    cases_phen2 = matched_df[matched_df[phen2]==2].nunique()
    unique_controls_phen2 = matched_df[matched_df[phen2]==1].nunique()
    sub_df = matched_df[[phen1, phen2]].dropna()
    if sub_df[phen1].nunique() == 2 and sub_df[phen2].nunique() == 2:
        contingency = pd.crosstab(sub_df[phen1], sub_df[phen2])
        if contingency.shape == (2, 2):
            if method=="original":
                print(contingency)
                #contingency = contingency*((cases_phen2*2)/(unique_controls_phen2+cases_phen2))
                #contingency = contingency.applymap(np.ceil)
                num_cases = sub_df[sub_df[phen1] == 2].shape[0]
                controls_sampled = sub_df[sub_df[phen1] == 1].sample(n=num_cases, random_state=42)
                cases = sub_df[sub_df[phen1] == 2]
                balanced_df = pd.concat([cases, controls_sampled])
                contingency = pd.crosstab(balanced_df[phen1], balanced_df[phen2])
                print(sum(contingency))
            #print((contingency.iloc[1,1]/contingency.iloc[1,0])/(contingency.iloc[0,1]/contingency.iloc[0,0]))
            print(contingency)
            OR, p = fisher_exact(contingency)
            print(OR)
            print(p)
            t2 = Table2x2(contingency)
            low, upp = t2.oddsratio_confint()

            return OR, p, cases_phen2, unique_controls_phen2 , contingency.iloc[1,1],contingency.iloc[1,0],contingency.iloc[0,1],contingency.iloc[0,0],low,upp
    return None, None,None, None,None, None,None, None, None, None


#####################################
#################main################
#####################################

pheno_1 = sys.argv[1]
pheno_2 = sys.argv[2]
input_data = sys.argv[3]
output_path = sys.argv[4]

#1. loading the data

starting_data_for_analysis = pd.read_csv(input_data)
print(starting_data_for_analysis.head())
covariate_cols = [col for col in starting_data_for_analysis.columns if ((col != "Participant ID") &  (col != "Ethnic_group_classes")&  (col != pheno_1)&  (col != pheno_2)&  (col != f"{pheno_1}_PRS"))]
print(covariate_cols)

#2. Calculate propensity score based on classical covariances.

# Calculate the propensity scores using the covariates
propensity_scores = calculate_propensity_scores(starting_data_for_analysis, pheno_1, covariate_cols)

#3. Calculate propensity score based on classical + PRS.
covariate_cols_with_PRS = covariate_cols
print(covariate_cols_with_PRS)

covariate_cols_with_PRS.append(f"{pheno_1}_PRS")

print(covariate_cols_with_PRS)

propensity_score_with_PRS = calculate_propensity_scores(starting_data_for_analysis, pheno_1, covariate_cols_with_PRS)

# Add the propensity scores to the dataframe
starting_data_for_analysis[f"{pheno_1}_propensity_score"] = propensity_scores
starting_data_for_analysis[f"{pheno_1}_propensity_score_with_PRS"] = propensity_score_with_PRS

# --- Save the dataframe with propensity scores ---
output_path = output_path +"/"+"_".join([pheno_1,"to",pheno_2])+"/"
if not os.path.exists(output_path):
    os.mkdir(output_path)
propensity_scores_path = output_path+'_'.join([pheno_1,"to",pheno_2,"with_propensity_scores_PRS_class_and_combined_propensity_score.csv"])
starting_data_for_analysis.to_csv(propensity_scores_path, index=False)
print(f"Data with propensity scores saved to: {propensity_scores_path}")

print(starting_data_for_analysis.head())


#4. Plot density plot and calculate SMD for cases amd control in the outcome disease and in the control diseases.

plot_multiple_ps(
    starting_data_for_analysis,
    ps_cols=[f"{pheno_1}_propensity_score", f"{pheno_1}_PRS" , f"{pheno_1}_propensity_score_with_PRS"],
    exposure_col=pheno_1,
    outcome_col=pheno_2,
    output_path=output_path+f"/{pheno_1}_{pheno_2}_propensity_score_density_plots.png"
)


############ Up to now we compleated calculating propensity score with and without genetics.
############ Now we needs to match the data based on the propensity scores

#5. Do matching for cases and controls based on each PS (PRS, classical, and combined)
results = []

for method, df_match, cases_phen1, unique_controls_phen1 in [
    ("original", starting_data_for_analysis , starting_data_for_analysis[starting_data_for_analysis[pheno_1] == 2], starting_data_for_analysis[starting_data_for_analysis[pheno_1] == 1]),
            ("classical_PSM", *match_by_score(starting_data_for_analysis, pheno_1, f"{pheno_1}_propensity_score",output_path=output_path+"matched_data_classical.csv")),
            ("PRS_PSM", *match_by_score(starting_data_for_analysis, pheno_1, f"{pheno_1}_PRS",output_path=output_path+"matched_data_PRS_PSM.csv")),
            ("both_PSM", *match_by_score(starting_data_for_analysis, pheno_1, f"{pheno_1}_propensity_score_with_PRS",output_path=output_path+"matched_data_combined_PSM.csv")),  ]:
#'''

            print(method)
            OR, p ,cases_phen2, unique_controls_phen2 ,yes_yes, yes_no, no_yes, no_no, CI_low, CI_high= run_fisher(df_match, pheno_1, pheno_2,method)
            if OR is not None:
                results.append([pheno_1, pheno_2, method, OR, p , cases_phen1, unique_controls_phen1 ,cases_phen2, unique_controls_phen2,yes_yes, yes_no, no_yes, no_no, CI_low, CI_high])
                print("results")
                print("results")
            else:
                print(f"Skipping {pheno_1} vs {pheno_2} ({method}): invalid contingency table")


# Save
results_df = pd.DataFrame(results, columns=["Phenotype 1", "Phenotype 2", "Matching Method", "Odds Ratio", "P-value","mun_of_cases_phen1","num_of_unique_controls_phen1","mun_of_cases_phen2","num_of_unique_controls_phen2","yes_yes", "yes_no", "no_yes", "no_no","CI_low", "CI_high"])
results_df.to_csv(output_path+"analysis_results.csv", index=False)

print("Analysis complete. Results saved to:")
print(output_path)

'''
progect_path = "/sci/archive/michall/hadasak/Hakcathon_FH/data/PRS_data/"

output_path = progect_path +"/"+"_".join([pheno_1,"to",pheno_2,"eroup_only"])+"/"

match_class_df = pd.read_csv(output_path+"matched_data_classical.csv")
match_PRS_df = pd.read_csv(output_path+"matched_data_PRS_PSM.csv")
match_combined_df = pd.read_csv(output_path+"matched_data_combined_PSM.csv")
results = []



for method, df_match, cases_phen1, unique_controls_phen1 in [
    ("original", starting_data_for_analysis , starting_data_for_analysis[starting_data_for_analysis[pheno_1] == 2], starting_data_for_analysis[starting_data_for_analysis[pheno_1] == 1]),
            ("classical_PSM",match_class_df, match_class_df[match_class_df[pheno_1] == 2],match_class_df[match_class_df[pheno_1] == 1]),
            ("PRS_PSM", match_PRS_df, match_PRS_df[match_PRS_df[pheno_1] == 2],match_PRS_df[match_PRS_df[pheno_1] == 1]),
            ("both_PSM", match_combined_df, match_combined_df[match_combined_df[pheno_1] == 2],match_combined_df[match_combined_df[pheno_1] == 1]),  ]:

'''