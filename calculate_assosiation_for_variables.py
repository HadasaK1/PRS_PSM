import pandas as pd
from scipy.stats import ttest_ind, fisher_exact, pearsonr
import numpy as np

# --- Load data ---

icd10_codes = ('E10', 'E11', 'G30', 'J45', 'K90', 'K50', 'I10' , 'C43' , 'G35' , 'M81' , 'G20' , 'F20')
progect_path = "/sci/archive/michall/hadasak/Hakcathon_FH/data/PRS_data/"
covariates_path = progect_path+"data_participant_cov_for_PRS_PSM_numbers_only.csv"  # Update paths accordingly
prs_path = progect_path+"Participant_table_PRS.csv"
phenotypes_path = progect_path + '_'.join(map(str, icd10_codes)) + "_phenotype_file.csv"
output_path = progect_path+'_'.join(map(str, icd10_codes)) + "association_matrix.csv"

covariates = pd.read_csv(covariates_path)
prs_scores = pd.read_csv(prs_path)
phenotypes = pd.read_csv(phenotypes_path,sep="\t")

print(covariates.head())
print(prs_scores.head())
print(phenotypes.head())

# --- Merge by ID ---
merged = covariates.merge(prs_scores, on="Participant ID").merge(phenotypes, left_on="Participant ID",right_on="FID")
print(merged.head())

# --- Get list of columns ---
covariate_cols = [col for col in covariates.columns if col != "Participant ID"]
prs_cols = [col for col in prs_scores.columns if col != "Participant ID"]
phenotype_cols = [col for col in phenotypes.columns if col != "FID"]
all_columns = covariate_cols + prs_cols + phenotype_cols


# --- Helper function to detect binary vs continuous ---
def is_binary(series):
    unique_vals = series.dropna().unique()
    return len(unique_vals) == 2


def is_continuous(series):
    return np.issubdtype(series.dtype, np.number)


# --- Initialize result list ---
results = []

# --- Run tests for all pairs of variables ---
for var1 in all_columns:
    for var2 in all_columns:
        if var1 == var2:  # Skip comparing the variable with itself
            continue

        data1 = merged[var1]
        data2 = merged[var2]

        # Skip if there are too many missing values
        if data1.isnull().sum() / len(data1) > 0.5 or data2.isnull().sum() / len(data2) > 0.5:
            continue

        try:
            # Binary vs Binary (Fisher's Exact Test)
            if is_binary(data1) and is_binary(data2):
                contingency_table = pd.crosstab(data1, data2)
                stat, p = fisher_exact(contingency_table)
                test = "Fisher's Exact Test"
                results.append([var1, var2, test, stat, p])

            # Continuous vs Continuous (Pearson Correlation)
            elif is_continuous(data1) and is_continuous(data2):
                corr, p = pearsonr(data1.dropna(), data2.dropna())
                test = "Pearson Correlation"
                results.append([var1, var2, test, corr, p])

            # Binary vs Continuous (T-test)
            elif (is_binary(data1) and is_continuous(data2)) or (is_binary(data2) and is_continuous(data1)):
                if is_binary(data1):
                    binary_data = data1
                    continuous_data = data2
                else:
                    binary_data = data2
                    continuous_data = data1

                unique_groups = binary_data.unique()

                if len(unique_groups) == 2:
                    # For binary categorical columns like 'yes'/'no' or '0'/ '1'
                    if binary_data.dtype == 'object':  # check if it's a string column
                        binary_data = binary_data.map({unique_groups[0]: 0, unique_groups[1]: 1})
                    else:
                        # If the column is already numeric, we assume it's in [0, 1] or [1, 2] format
                        pass
                group1 = continuous_data[binary_data == 1]
                group2 = continuous_data[binary_data != 1]

                print(group1.shape)
                print(group2.shape)

                stat, p = ttest_ind(group1.dropna(), group2.dropna(), equal_var=False)
                test = "T-test"
                results.append([var1, var2, test, stat, p])

        except Exception as e:
            print(f"Skipping {var1} vs {var2} due to error: {e}")

# --- Save results to CSV ---
results_df = pd.DataFrame(results, columns=["Variable 1", "Variable 2", "Test", "Statistic", "P-value"])
results_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")