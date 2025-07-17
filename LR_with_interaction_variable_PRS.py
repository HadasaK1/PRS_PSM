import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


icd10_codes = ['E10', 'E11', 'G30', 'J45', 'K90', 'K50', 'I10' , 'C43' , 'G35' , 'M81' , 'G20' , 'F20']

# Paths
proj_path = "/sci/archive/michall/hadasak/Hakcathon_FH/data/PRS_data/"
input_path = proj_path + '_'.join(map(str, icd10_codes)) +"merged_with_propensity_scores_eroup_only_both_PRS_and_cov.csv"
output_path = proj_path + '_'.join(map(str, icd10_codes)) +"_logistic_model_coefficients_for_PRS.csv"

# Load data
df = pd.read_csv(input_path)
df[icd10_codes] = df[icd10_codes].replace({1: 0, 2: 1})

print(df.columns)

# Diseases of interest


disease_columns = icd10_codes

# Generate corresponding propensity and PRS column names
propensity_cols = [f"{d}_propensity_score" for d in disease_columns]

# Main loop
all_coefficients = pd.DataFrame()

for i in range(len(disease_columns)):
    for j in range(len(disease_columns)):
        if i==j:
            continue
        phen1 = disease_columns[i]
        phen2 = disease_columns[j]

        relevant_column = [phen1, "age", "age_squred" , "pack_years_of_smoking", "BMI", "genetic_sex", "ever_smoke", phen2+"_PRS", phen1+"_PRS", phen2]
        df_for_model = df[[col for col in relevant_column if col in df.columns]]

        # Set target
        y = df_for_model[phen2]
        print(y)
        y = y.dropna()
        print(y.unique())

        # Set predictors: drop phen1 from the list of covariates
        X = df_for_model.drop(columns=[phen2])

        # Drop non-numeric columns if needed
        X = X.select_dtypes(include=['number'])

        # Scale the predictors
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Add constant for intercept (important for statsmodels)
        X_scaled = sm.add_constant(X_scaled)

        # Fit logistic regression model
        logit_model = sm.Logit(y, X_scaled)
        result = logit_model.fit(disp=0)  # disp=0 disables fitting output
        pvals = result.pvalues.values
        _, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')

        # Save coefficients to DataFrame
        coef_df = pd.DataFrame({
            'outcome_phenotype': phen2,
            'exposure_phenotype': phen1,
            'variable': result.params.index,
            'coefficient': result.params.values,
            'p_value': result.pvalues.values,
            'FDR_corrected_p': pvals_fdr })

        # Append to main results
        all_coefficients = pd.concat([all_coefficients, coef_df], ignore_index=True)
        print(coef_df)

# Save
all_coefficients.to_csv(output_path, index=False)

print("Analysis complete. Results saved to:")
print(output_path)
