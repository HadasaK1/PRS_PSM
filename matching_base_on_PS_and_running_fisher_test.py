import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import fisher_exact
import math

icd10_codes = ['E10', 'E11', 'G30', 'J45', 'K90', 'K50', 'I10' , 'C43' , 'G35' , 'M81' , 'G20' , 'F20']

# Paths
proj_path = "/sci/archive/michall/hadasak/Hakcathon_FH/data/PRS_data/"
input_path = proj_path + '_'.join(map(str, icd10_codes)) +"merged_with_propensity_scores_eroup_only_both_PRS_and_cov.csv"
output_path = proj_path + '_'.join(map(str, icd10_codes)) +"causality_results_all_matching_methods_eroup_only_rand_undersamp.csv"

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

# Fisher test function
def run_fisher(matched_df, phen1, phen2,method):
    cases_phen2 = matched_df[matched_df[phen2]==2]["Participant ID"].nunique()
    unique_controls_phen2 = matched_df["Participant ID"][matched_df[phen2]==1].nunique()
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
            return OR, p, cases_phen2, unique_controls_phen2 , contingency.iloc[1,1],contingency.iloc[1,0],contingency.iloc[0,1],contingency.iloc[0,0]
    return None, None,None, None,None, None,None, None

# Main loop
results = []

for i in range(len(disease_columns)):
    for j in range(len(disease_columns)):
        if i==j:
            continue
        phen1 = disease_columns[i]
        phen2 = disease_columns[j]

        for method, df_match, cases_phen1, unique_controls_phen1 in [
            ("original", df , df[df[phen1] == 2]["Participant ID"].nunique(), df[df[phen1] == 1]["Participant ID"].nunique()),
            ("classical_PSM", *match_by_score(df, phen1, f"{phen1}_propensity_score")),
            ("PRS_PSM", *match_by_score(df, phen1, f"{phen1}_PRS")),
            ("both_PSM", *match_by_score(df, phen1, f"{phen1}_propensity_score_with_PRS")),  ]:

            print(method)
            OR, p ,cases_phen2, unique_controls_phen2 ,yes_yes, yes_no, no_yes, no_no= run_fisher(df_match, phen1, phen2,method)
            if OR is not None:
                results.append([phen1, phen2, method, OR, p , cases_phen1, unique_controls_phen1 ,cases_phen2, unique_controls_phen2,yes_yes, yes_no, no_yes, no_no])
            else:
                print(f"Skipping {phen1} vs {phen2} ({method}): invalid contingency table")

        #df_psm1, _1 , _2 = match_by_score(df, phen1, f"{phen1}_propensity_score")
        #df_psm2, cases_phen1, unique_controls_phen1 = match_by_score(df_psm1, phen1, f"{phen1}_PRS")
        #OR, p , cases_phen2, unique_controls_phen2 = run_fisher(df_psm2, phen1, phen2,"both_PSM")
        #if OR is not None:
        #    results.append([phen1, phen2, "both_PSM", OR, p,cases_phen1, unique_controls_phen1 ,cases_phen2, unique_controls_phen2])
        #else:
        #    print(f"  Skipped (both_PSM)")

# Save
results_df = pd.DataFrame(results, columns=["Phenotype 1", "Phenotype 2", "Matching Method", "Odds Ratio", "P-value","mun_of_cases_phen1","num_of_unique_controls_phen1","mun_of_cases_phen2","num_of_unique_controls_phen2","yes_yes", "yes_no", "no_yes", "no_no"])
results_df.to_csv(output_path, index=False)

print("Analysis complete. Results saved to:")
print(output_path)
