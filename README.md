# **PRS-PSM: Polygenic Risk Score Propensity Score Matching**

**PRS-PSM** is a Python-based tool for performing **propensity score matching (PSM)** to explore causal relationships between two diseases, **A (exposure)** and **B (outcome)**.

Unlike classical PSM that only balances clinical covariates, **PRS-PSM also balances genetic risk** by incorporating the **Polygenic Risk Score (PRS)** of the exposure disease.

This helps to **reduce genetic confounding** and provides a cleaner comparison between cases and controls.

---

## **Features**

* Matches individuals based on:

  * Classical covariates (e.g., age, sex, smoking status, BMI). You chose the relevant covariance based on your exposure and outcome phenotypes
  * Genetic risk for the exposure phenotype (PRS)
* Outputs matched datasets for downstream causal inference analysis.
* Simple command-line interface.
* Reproducible results with version-controlled dependencies.

---

## **Input Data Format**

You need to provide a CSV file with the following structure:

| participant\_id | sex | age | smoking | exposure\_pheno | exposure\_pheno\_PRS | outcome\_pheno |
| --------------- | --- | --- | ------- | --------------- | -------------------- | -------------- |
| 101             | 2   | 55  | 1       | 2               | 0.35                 | 1              |
| 102             | 1   | 60  | 0       | 1               | -0.12                | 1              |
| 103             | 2   | 58  | 1       | 2               | 0.58                 | 2              |

### **Important Notes**

* **Exposure phenotype (`exposure_pheno`)**

  * Must be coded as `1 = control` and `2 = case`.
* **Outcome phenotype (`outcome_pheno`)**

  * Must also be coded as `1 = control` and `2 = case`.
* **PRS column** should be named exactly as `[exposure_pheno]_PRS`.
* No missing values (`NaN`) are allowed.
* All values must be numeric.

---

## **Installation**

### **Step 1: Download the Code**

Clone the repository or download the files directly:

```bash
git clone https://github.com/YourUsername/PRS_PSM.git
cd PRS_PSM
```

Make sure you have:

* `PRS_PSM.py` (main script)
* `requirements.txt` (dependencies list)

---

### **Step 2: Install Required Packages**

Install dependencies using:

```bash
pip install -r requirements.txt
```

> 💡 *Tip: It’s recommended to use a virtual environment (e.g., `venv` or `conda`) to avoid conflicts with other projects.*

---

## **Usage**

Run the script using:

```bash
python PRS_PSM.py [exposure_pheno] [outcome_pheno] [input_csv_file] [output_path]
```

### **Example**

Suppose:

* Exposure phenotype: `Diabetes`
* Outcome phenotype: `HeartDisease`
* Input file: `data/input_dataset.csv`
* Output folder: `results/`

Run:

```bash
python PRS_PSM.py Diabetes HeartDisease data/input_dataset.csv results/
```

This will generate matched datasets and save them in the `results/` directory.

---

## **Output**

The script will produce:

* A CSV file containing the **matched dataset**, including:

  * Original columns
  * Matching identifiers
  * Density plots for the propensity score in the cases and controls for the exposure and outcome phenotypes.
  * Logs and summary statistics for matching performance.



## **Recommended Workflow**

1. **Prepare input data** → Clean data, ensure correct coding and naming.
2. **Run PRS-PSM** → Generate matched datasets.
3. **Evaluate balance** → Check that covariates and PRS are well balanced.
4. **Perform causal analysis** → Use matched data for downstream statistical testing.

---

## **Troubleshooting**

* **Issue:** Script crashes due to missing values
  **Solution:** Check your CSV file for `NaN` and clean it before running.

* **Issue:** PRS column not recognized
  **Solution:** Make sure the PRS column name matches exactly `[exposure_pheno]_PRS`.

* **Issue:** Package conflicts
  **Solution:** Install in a fresh virtual environment.

---

