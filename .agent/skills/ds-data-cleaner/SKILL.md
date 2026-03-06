---
name: ds-data-cleaner
description: Specialized skill for data auditing, cleaning, and preprocessing. Use for missing value handling, outlier detection, and feature encoding.
risk: medium
source: gemini
version: 1.0.0
date_added: '2026-03-05'
---

# Data Cleaning Protocol

## Instructions
1. **Initial Audit**: Before modifying data, run a `df.info()` and `df.describe()` to establish a baseline. Generate a summary artifact of null counts and data types.
2. **Handling Missing Values**:
   - For **Categorical**: Default to "Unknown" or Mode.
   - For **Numerical**: Check skewness. Use Median for skewed data; Mean for normal distributions.
3. **Outlier Strategy**: Identify outliers using the IQR method (1.5 * IQR). Do not drop them unless they are clearly sensor errors; prefer Winsorization or log-transformation.
4. **Consistency**: Standardize column names to `snake_case`. Convert date columns to `datetime64` objects.
5. **Validation**: Every cleaning script must end with an assertion check (e.g., `assert df['col'].isnull().sum() == 0`).

## Tools & Libraries
- Primary: `pandas`, `numpy`, `polars`
- Visualization: `missingno` (for null patterns), `seaborn` (for distributions)

## Example Interaction
"Use @ds-data-cleaner to audit `raw_leads.csv` and suggest a strategy for the 15% missing 'Phone' values."