import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import fisher_exact
import math
import matplotlib.pyplot as plt
import seaborn as sns

icd10_codes = ['E10', 'E11', 'G30', 'J45', 'K90', 'K50', 'I10' , 'C43' , 'G35' , 'M81' , 'G20' , 'F20']

# Paths
proj_path = "/sci/archive/michall/hadasak/Hakcathon_FH/data/PRS_data/"
input_path = proj_path + '_'.join(map(str, icd10_codes)) +"merged_with_propensity_scores_eroup_only_both_PRS_and_cov.csv"
output_path = proj_path +'/matched_data/'

# Load data
df = pd.read_csv(input_path)

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
    for col in df.columns
}
df.rename(columns=new_columns, inplace=True)

print(df.columns)

# Diseases of interest


disease_columns = icd10_codes

# Generate corresponding propensity and PRS column names
propensity_cols = [f"{d}_propensity_score" for d in disease_columns]

# Matching function
def match_by_score(df, disease_col, score_col):
    treated = df[(df[disease_col] == 2) & (~df[score_col].isna())].copy()
    control = df[(df[disease_col] == 1) & (~df[score_col].isna())].copy()

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
    cases_phen1 = matched_df[matched_df[disease_col] == 2]["Participant ID"].nunique()
    unique_controls_phen1 = matched_df[matched_df[disease_col] == 1]["Participant ID"].nunique()


    return matched_df, cases_phen1, unique_controls_phen1

def compute_smd(df, treatment_col, covariates):
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    smd = {}
    for cov in covariates:
        if df[cov].dtype == 'object' or len(df[cov].unique()) <= 2:  # categorical
            p1 = treated[cov].value_counts(normalize=True).reindex([1], fill_value=0).values[0]
            p0 = control[cov].value_counts(normalize=True).reindex([1], fill_value=0).values[0]
            smd[cov] = np.abs(p1 - p0)
        else:  # continuous
            mean_treated = treated[cov].mean()
            mean_control = control[cov].mean()
            sd_pooled = np.sqrt((treated[cov].var() + control[cov].var()) / 2)
            smd[cov] = np.abs(mean_treated - mean_control) / sd_pooled
    return smd

def plot_covariate_distributions(icd_list, matched_dict, method_name, covariates, treatment_col='treatment'):
    num_icds = len(icd_list)
    for cov in covariates:
        fig, axes = plt.subplots(1, num_icds, figsize=(5*num_icds, 4), sharey=True)
        if num_icds == 1:
            axes = [axes]
        for i, icd in enumerate(icd_list):
            df = matched_dict[icd]
            smd = compute_smd(df, treatment_col, [cov])[cov]
            ax = axes[i]
            if df[cov].dtype == 'object' or len(df[cov].unique()) <= 2:
                # Bar plot for categorical
                df_plot = df[[cov, treatment_col]].copy()
                df_plot[cov] = df_plot[cov].astype(str)
                sns.barplot(data=df_plot, x=cov, y=[1]*len(df_plot), hue=treatment_col, estimator=lambda x: len(x)/len(df_plot), ax=ax)
                ax.set_ylabel("Proportion")
            else:
                # Density plot for continuous
                sns.kdeplot(data=df, x=cov, hue=treatment_col, fill=True, common_norm=False, alpha=0.5, ax=ax)
                ax.set_ylabel("Density")
            ax.set_title(f"{method_name} | ICD: {icd}\nSMD={smd:.3f}")
            ax.set_xlabel(cov)
        plt.tight_layout()
        plt.suptitle(f"{cov} - {method_name}", fontsize=16, y=1.02)
        plt.show()

icd10_codes = ['E10', 'E11', 'G30', 'J45', 'K90', 'K50', 'I10' , 'C43' , 'G35' , 'M81' , 'G20' , 'F20']

matched_data_dict = {}  # structure: matched_data_dict[phen1][method] = df_match

for icd10 in icd10_codes:
    for method, df_match, cases_phen1, unique_controls_phen1 in [
            ("original", df , df[df[phen1] == 2]["Participant ID"].nunique(), df[df[phen1] == 1]["Participant ID"].nunique()),
            ("classical_PSM", *match_by_score(df, phen1, f"{phen1}_propensity_score")),
            ("PRS_PSM", *match_by_score(df, phen1, f"{phen1}_PRS")),
            ("both_PSM", *match_by_score(df, phen1, f"{phen1}_propensity_score_with_PRS")),  ]:

        df_match.to_csv(output_path+"_".join(method,icd10)+"_matched.csv",index=False)

        if phen1 not in matched_data_dict:
            matched_data_dict[phen1] = {}
            matched_data_dict[phen1][method] = df_match

matched_classical_dict = {icd: methods['classical_PSM'] for icd, methods in matched_data_dict.items() if
                                  'classical_PSM' in methods}
matched_prs_dict = {icd: methods['PRS_PSM'] for icd, methods in matched_data_dict.items() if
                            'PRS_PSM' in methods}
matched_combined_dict = {icd: methods['both_PSM'] for icd, methods in matched_data_dict.items() if
                                 'both_PSM' in methods}

plot_covariate_distributions_per_method(icd_list=matched_classical_dict.keys(), matched_data_dict=matched_classical_dict, method_name='Classical', covariates=all_covs)
plot_covariate_distributions_per_method(icd_list=matched_prs_dict.keys(), matched_data_dict=matched_prs_dict, method_name='PRS', covariates=all_covs)
plot_covariate_distributions_per_method(icd_list=matched_combined_dict.keys(), matched_data_dict=matched_combined_dict, method_name='Combined', covariates=all_covs)

        # Example usage:
# Suppose you have dictionaries of matched data:
# matched_classical_dict = {'I10': df1, 'E11': df2, ...}
# matched_prs_dict = ...
# matched_combined_dict = ...
# ICD codes:

# Define covariates
            #continuous_covs = ['age','age_squred','pack_years_of_smoking','BMI',icd10+'_PRS']
            #categorical_covs = ['genetic_sex', 'ever_smoke']
            #all_covs = continuous_covs + categorical_covs

# Run for each matching method
#plot_covariate_distributions(icd_list, df_match, , all_covs)

print("Analysis complete. Results saved to:")
print(output_path)
