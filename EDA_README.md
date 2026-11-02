<div align="center">

# Indian Startup Funding: Exploratory Data Analysis

**SWYNEX Technologies · Data Analyst Internship · Task 2**

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Analysis-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisation-11557C)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-2EA44F)

*A cleaned startup-funding table audited for outliers, corrected where evidence exists, and analysed by year, industry, city, investor, funding stage and startup.*

</div>

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Repository Structure](#3-repository-structure)
4. [Data Audit (Before Analysis)](#4-data-audit-before-analysis)
5. [Analysis Workflow](#5-analysis-workflow)
6. [Key Findings](#6-key-findings)
7. [Before vs After](#7-before-vs-after)
8. [Final Data Dictionary](#8-final-data-dictionary)
9. [Decisions & Assumptions](#9-decisions--assumptions)
10. [Limitations](#10-limitations)
11. [How to Reproduce](#11-how-to-reproduce)
12. [Key Learnings](#12-key-learnings)
13. [Next Steps](#13-next-steps)
14. [Author](#14-author)

---

## 1. Project Overview

**Objective:** Use the cleaned dataset from Task 1 to understand Indian startup funding: how funding changed over time, which industries, cities, investors and funding stages matter most, and which startups raised the most money.

**Approach:** The analysis was done in a Google Colab notebook with `pandas` and `matplotlib`. Every section follows the same pattern: **check the data first, analyse second, state the findings and caveats third.** Each section also names the next step, so the notebook reads as one connected argument.

**Outcome:**
- **11 wrong funding amounts** (rupee values stored as USD, plus data-entry errors) were found through outlier detection, checked against public news reports and corrected. This removed about **$5.4B** of inflated funding from the total.
- **Six analyses** (year, industry, city, investor, funding stage, startup) with findings and caveats written next to each result.
- An **analysis-ready table of 3,044 rows × 16 columns** (`startup_funding_eda.csv`) that keeps both the corrected and the original amount, ready for the dashboard stage.

---

## 2. Dataset

| Property | Detail |
|---|---|
| Source | Output of Task 1 (Kaggle Indian startup funding dataset, cleaned) |
| Input file | `cleaned_startup_funding.csv` |
| Input size | 3,044 rows × 10 columns |
| Output file | `startup_funding_eda.csv` |
| Output size | 3,044 rows × 16 columns |
| Period covered | 2 January 2015 to 13 January 2020 |
| Content | Funding rounds of Indian startups: date, company, industry, city, investors, round type, amount in USD |

---

## 3. Repository Structure

```
SWYNEX-Startup-Funding-EDA/
├── cleaned_startup_funding.csv                              # Input: Task 1 output
├── startup_funding_eda.csv                                  # Final analysis-ready data (16 columns)
├── Indian_Startup_Funding_Exploratory_Data_Analysis.ipynb   # Full notebook with every step
└── README.md                                                # This document
```

---

## 4. Data Audit (Before Analysis)

Checked using `df.shape`, `df.info()`, `df.isna().sum()`, `(df == 'Unknown').sum()`, `df.duplicated().sum()` and `df.describe()`.

| # | Check | Result |
|---|---|---|
| 1 | Shape | 3,044 rows × 10 columns |
| 2 | Data types | `date` is `datetime64`, `amount_usd` is `float64`, `sr_no` is `int64`, the rest are text |
| 3 | Missing values | Only `amount_usd` has real missing values: **971 rows (31.9%)**, because the amount was never disclosed |
| 4 | `Unknown` placeholder labels | `industry` 171 · `city` 180 · `investment_type` 4 (placeholders from Task 1, kept and excluded per analysis) |
| 5 | Empty strings | 0 in every text column |
| 6 | Duplicate rows | **0** (checked after dropping `sr_no`, which is unique by design) |
| 7 | Skewed amounts | Mean ≈ **$18.4M** against a median of **$1.75M**; maximum **$3.9B**. A few extreme values pull the average far above the typical deal |

**Why the last row matters most:** the gap between mean and median is the first sign that some amounts are wrong or extreme. That is what triggers the outlier work in Section 5.2.

---

## 5. Analysis Workflow

### Step 1: Load and audit
```python
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/cleaned_startup_funding.csv', parse_dates=['date'])

print(df.shape)
df.info()
print(df.isna().sum())
print((df == 'Unknown').sum())
print('Duplicates:', df.drop(columns='sr_no').duplicated().sum())
df.describe()
```
`parse_dates` restores the `date` column type, because a CSV file does not store data types. Duplicates are checked without `sr_no`, since a serial number would make every row look unique.

### Step 2: Outlier detection and correction

This is the most important step of the project, because wrong amounts change every total that follows.

**2.1 Spot-check the largest values.** The 40 largest amounts were listed, and every deal above $100M was flagged as `suspect` (**56 rows**). Seed/Angel deals of $50M or more were listed separately, because a seed round that large is very unlikely: Digital Mall Asia ($220M), FreshVnF ($140M), Shape ($100M), Rein Games ($50M) and Dhruva Space ($50M). `suspect` is a "look here" list only; it also contains real large deals such as Flipkart and Udaan.

**2.2 Outliers within each investment type.** A normal deal size differs by round, so a flat cut-off is unfair. Amounts above `Q3 + 3 × IQR` were flagged **within each investment type**. The multiplier 3 (instead of the usual 1.5) keeps only extreme values.
```python
def flag(g):
    q1, q3 = g.quantile(.25), g.quantile(.75)
    return g > q3 + 3*(q3 - q1)

df['outlier'] = df.groupby('investment_type').amount_usd.transform(flag)
```
Result: **118 rows flagged** (Private Equity 82, Seed/Angel 28, Series B 3, Other 1, Pre-Series A 1, Series A 1, Series C 1, Series D 1).

**2.3 Manual review.** Flagged rows were reviewed by hand, round by round:

| Round | Flagged | Decision |
|---|---|---|
| Seed/Angel | 28 | The top 5 ($50M to $220M) are far too large for a seed round. The other 23 ($2.4M to $12M) are high but possible, so they were left unchanged |
| Series B | 3 | Vogo Automotive ($283M) and Rapido Bike Taxi ($3.9B) look unrealistic. Automation Anywhere ($300M) is large but possible |
| Other rounds | 5 | GOQii, Recykal and Kuvera were checked against 2019 news reports and marked for verification. Udaan ($585M) and Paytm ($1B) match news reports and were left unchanged |

**2.4 Mark rows to verify.** Ten rows with unrealistic amounts were marked `to_verify`.

**2.5 Fix the wrong amounts.** Each of the 10 rows was checked against public news reports from 2018 and 2019. Most were **rupee amounts stored as USD** (for example, Rs 22 Cr shown as $220M); Recykal and Kuvera were data-entry errors. The corrected USD value goes into `amount_usd`, and the original value is preserved in `amount_original`.

| Startup | Original amount | Corrected amount | Reason |
|---|---:|---:|---|
| Rapido Bike Taxi | $3,900,000,000 | $54,000,000 | Rs 390 Cr |
| Vogo Automotive | $283,000,000 | $4,000,000 | Rs 28.3 Cr |
| GOQii | $450,000,000 | $6,240,000 | Rs 45 Cr |
| Digital Mall Asia | $220,000,000 | $3,060,000 | Rs 22 Cr |
| FreshVnF | $140,000,000 | $2,000,000 | $2M reported |
| Shape | $100,000,000 | $200,000 | $200K reported |
| Dhruva Space | $50,000,000 | $700,000 | Rs 5 Cr |
| Rein Games | $50,000,000 | $700,000 | Rs 5 Cr |
| Kuvera | $45,000,000 | $4,500,000 | $4.5M reported (data-entry error) |
| Recykal | $26,000,000 | $2,000,000 | $2M reported (data-entry error) |
| Furtados School of Music | $200,000,000 | $2,800,000 | Rs 20 Cr (not flagged by the IQR rule because its group is tiny, so it was corrected by name) |

```python
if 'amount_original' not in df.columns:      # keep the original only on the first run
    df['amount_original'] = df.amount_usd

m = df.to_verify
new = df.loc[m, 'startup_name'].map(fix)
assert new.notna().all(), 'a to_verify row has no fix'
df.loc[m, 'amount_usd'] = new
```
The `assert` stops the notebook if a marked row has no correction, so nothing is changed silently. The 11 corrected rows carried about **$5.46B before** the fix and about **$0.08B after** it.

**2.6 Re-check after the fixes.** The same IQR rule was run again and saved in a new column `outlier_after_fix`, so the first flags stay visible. **110 rows** remain flagged. Seed/Angel fell from 28 to 24 and Series B from 3 to 1. One row is new: Engineer.ai (Series A, $29.5M) is above the limit only because the Series A threshold dropped after Kuvera was corrected, so it was left unchanged. The 82 Private Equity rows are a mixed group that includes real large deals such as Flipkart and Paytm, so they were also left unchanged.

### Step 3: Year-wise trend
```python
df['year'] = df.date.dt.year
yr = df.groupby('year').agg(deals=('sr_no','count'),
                            with_amount=('amount_usd','count'),
                            total=('amount_usd','sum'),
                            median=('amount_usd','median'))
yr['missing_pct'] = (1 - yr.with_amount/yr.deals) * 100
```
The **median** is used next to the total because a few very large deals pull the total up. The share of deals with a missing amount is shown for each year so the reader can judge how complete each total is.

| Year | Deals | With amount | Total funding | Median deal | Amount missing |
|---|---:|---:|---:|---:|---:|
| 2015 | 936 | 654 | $8.67B | $1.50M | 30% |
| 2016 | 993 | 586 | $3.83B | $1.00M | 41% |
| 2017 | 687 | 456 | $10.43B | $2.25M | 34% |
| 2018 | 310 | 265 | $5.02B | $3.80M | 15% |
| 2019 | 111 | 105 | $4.42B | $6.24M | 5% |
| 2020 (Jan only) | 7 | 7 | $0.39B | $9.00M | 0% |

A three-panel chart shows deals, total funding and median deal size side by side.

### Step 4: Industry-wise funding
Deals are grouped by `industry`. Only industries with **at least 10 deals that have an amount** are ranked, so that the median is reliable and one-off labels do not distort the list. `Unknown` industries are excluded.

| Industry | Deals | With amount | Total funding | Median deal |
|---|---:|---:|---:|---:|
| eCommerce | 299 | 205 | $8.04B | $3.00M |
| Consumer Internet | 941 | 589 | $6.25B | $1.00M |
| Technology | 489 | 321 | $2.31B | $1.50M |
| Finance | 62 | 57 | $1.93B | $6.00M |
| FinTech | 16 | 16 | $1.36B | $10.00M |
| Food & Beverage | 35 | 25 | $0.37B | $2.00M |
| Healthcare | 70 | 44 | $0.28B | $2.10M |
| Logistics | 32 | 23 | $0.24B | $6.00M |
| EdTech | 20 | 18 | $0.19B | $0.89M |
| Education | 24 | 19 | $0.10B | $1.50M |

### Step 5: City-wise funding
City labels were first checked for spelling variants, because variants would split one city into several rows. Clear typos, neighbourhoods and codes were mapped to their city in a **new column** `city_clean`; the original `city` column is untouched.
```python
city_fix = {'Kormangala':'Bengaluru', 'Andheri':'Mumbai', 'Chembur':'Mumbai',
            'Taramani':'Chennai', 'Nw Delhi':'New Delhi', 'Kolkatta':'Kolkata',
            'Ahemdabad':'Ahmedabad', 'Bhubneswar':'Bhubaneswar',
            'San Jose,':'San Jose', 'SFO':'San Francisco', 'US':'USA'}
df['city_clean'] = df.city.replace(city_fix)      # 13 rows changed
```
Labels that hold two cities (`Bengaluru and Gurugram`) or only a state (`Karnataka`) were left as they are, because the intended city cannot be known.

| City | Deals | With amount | Total funding | Median deal |
|---|---:|---:|---:|---:|
| Bengaluru | 851 | 593 | $14.60B | $2.20M |
| Mumbai | 575 | 409 | $4.83B | $2.00M |
| Gurugram | 342 | 244 | $3.88B | $2.50M |
| New Delhi | 463 | 270 | $3.15B | $1.00M |
| Noida | 94 | 57 | $1.23B | $1.00M |
| Chennai | 99 | 76 | $0.76B | $2.50M |
| Pune | 112 | 77 | $0.62B | $1.30M |
| Hyderabad | 100 | 73 | $0.38B | $1.00M |
| Jaipur | 30 | 14 | $0.15B | $1.00M |
| Ahmedabad | 41 | 29 | $0.12B | $1.00M |

**Delhi NCR view.** `Delhi NCR` is not a label in the data, so it was built as the sum of New Delhi, Gurugram and Noida: **899 deals** (571 with an amount) raising about **$8.26B**. Other nearby cities such as Faridabad are not included.

### Step 6: Top investors
A deal can list several investors in one cell. The cell is split on commas so every investor gets credit for every deal they joined, and each investor is counted **once per deal**.
```python
tail = r'\s*(?:&|and)\s+(?:others?\b|existing\b|one more\b).*$'
inv['investor'] = inv.investor.str.replace(tail, '', regex=True, case=False).str.strip()
inv['investor'] = inv.investor.replace(inv_fix)     # merge name variants
inv = inv.drop_duplicates(['sr_no', 'investor'])    # once per deal
```
Cleaning rules: generic endings (`& Others`, `and existing investors`) are removed; clear variants are merged (`Sequoia India`, `Sequoia Capital India` and `Sequoia` into `Sequoia Capital`; likewise Accel, Tiger Global, IDG Ventures, Matrix Partners); generic labels such as `Undisclosed` are dropped.

| Rank | Investor | Deals |
|---:|---|---:|
| 1 | Sequoia Capital | 112 |
| 2 | Accel Partners | 84 |
| 3 | Tiger Global | 52 |
| 4 | Blume Ventures | 51 |
| 5 | Kalaari Capital | 51 |
| 6 | SAIF Partners | 50 |
| 7 | IDG Ventures | 46 |
| 8 | Indian Angel Network | 43 |
| 9 | Matrix Partners | 34 |
| 10 | Ratan Tata | 31 |

Merging name variants changed the ranking: Sequoia Capital rose from 72 to 112 deals and Tiger Global from 28 to 52.

### Step 7: Funding by investment type
`Unknown` investment types are excluded.

| Investment type | Deals | With amount | Total funding | Median deal |
|---|---:|---:|---:|---:|
| Seed/Angel | 1,542 | 864 | $0.50B | $0.36M |
| Private Equity | 1,367 | 1,082 | $27.40B | $6.00M |
| Debt | 31 | 30 | $0.19B | $4.90M |
| Series A | 24 | 22 | $0.16B | $4.00M |
| Series B | 21 | 21 | $0.68B | $15.11M |
| Series C | 14 | 14 | $0.60B | $44.04M |
| Series D | 12 | 12 | $1.48B | $65.00M |
| Other | 11 | 8 | $1.21B | $10.00M |
| Pre-Series A | 9 | 8 | $0.02B | $2.50M |
| Series E to J, Crowdfunding (combined) | 9 | 9 | $0.51B | not reliable |

Series E to J have only 1 or 2 deals each, so they are not compared. The single Series J row is Ola Cabs ($1M, July 2019), which is only a small part of a larger round; the amount was left unchanged because it does not match a reported figure. The median-by-stage chart therefore shows only **Seed/Angel and Series A to D**, and leaves out the mixed `Private Equity` group.

### Step 8: Top startups
The same startup can appear under several spellings, which would split its total across rows. Names that differ only by capital letters, spaces, punctuation or endings such as `.com` and `Pvt Ltd` were reduced to a key and mapped to the most common spelling in a **new column** `startup_clean`.
```python
def name_key(s):
    s = re.sub(r'\.com|\.in|\bpvt\b|\bltd\b|\bprivate\b|\blimited\b', '', s.lower())
    return re.sub(r'[^a-z0-9]', '', s)

canon = names.groupby('key').startup_name.agg(lambda x: x.value_counts().idxmax())
df['startup_clean'] = names.key.map(canon)
```
**189 startups** appeared under more than one spelling. Names with different words were merged by hand (`Ola Cabs` into `Ola`, `OYO Rooms` and `Oyo` into `OYO`). `Ola Electric`, `OYOfit` and `Paytm Marketplace` are different companies or business units and stay separate. Unique names fell from **2,457 to 2,242**.

`True North` ($600M) was excluded from the ranking because it is a private equity fund raise, not startup funding.

| Rank | Startup | Deals | With amount | Total funding |
|---:|---|---:|---:|---:|
| 1 | Flipkart | 6 | 6 | $4.76B |
| 2 | Paytm | 7 | 5 | $3.15B |
| 3 | Ola | 13 | 12 | $2.05B |
| 4 | OYO | 9 | 9 | $1.00B |
| 5 | Udaan | 4 | 4 | $0.87B |
| 6 | Snapdeal | 2 | 2 | $0.70B |
| 7 | Byju’s | 7 | 6 | $0.53B |
| 8 | BigBasket | 5 | 5 | $0.51B |
| 9 | Zomato | 4 | 4 | $0.44B |
| 10 | PolicyBazaar | 4 | 4 | $0.34B |

### Step 9: Export
```python
final_cols = ['sr_no', 'date', 'year', 'startup_name', 'startup_clean', 'industry',
              'subvertical', 'city', 'city_clean', 'investors', 'investment_type',
              'amount_usd', 'amount_original', 'amount_fixed', 'outlier_after_fix', 'remarks']
df[final_cols].to_csv('startup_funding_eda.csv', index=False)     # (3044, 16)
```
Helper columns used only during the review (`suspect`, `outlier`, `to_verify`) are not exported. `outlier_after_fix` is kept as the final flag.

---

## 6. Key Findings

| # | Finding |
|---|---|
| 1 | **Fewer, bigger deals.** Deals fell from 993 (2016) to 111 (2019), while the median deal size rose from $1.0M to $6.2M. Total funding peaked in 2017 ($10.4B) because of a few very large deals, then fell to about $5B in 2018 and 2019 |
| 2 | **Funding is concentrated by industry.** eCommerce ($8.0B) and Consumer Internet ($6.3B) hold about 68% of the top-10 industry funding. Consumer Internet has the most deals (941) but a median of only $1.0M, so it is many small deals. FinTech (median $10M), Finance ($6M) and Logistics ($6M) have fewer but larger deals |
| 3 | **Bengaluru leads by money, Delhi NCR by deal count.** Bengaluru raised about $14.6B, roughly half of the top-10 city total ($29.7B). Delhi NCR (New Delhi, Gurugram and Noida together) has more deals than Bengaluru (899 vs 851) but raised less ($8.3B vs $14.6B). Median deals are $1.0M to $2.5M in all ten cities, so Bengaluru's lead comes mostly from a few very large deals, not from a higher typical deal |
| 4 | **Stage drives deal size.** The median rises from $0.36M (Seed/Angel) to $4.0M (Series A), $15.1M (Series B), $44.0M (Series C) and $65.0M (Series D). Private Equity holds about 84% of total funding ($27.4B of $32.8B); Seed/Angel has the most deals (1,542) but only about 1.5% of the money |
| 5 | **Most active investors.** Sequoia Capital (112 deals) and Accel Partners (84) lead, well ahead of Tiger Global (52). Individual angels such as Ratan Tata (31) rank alongside venture funds |
| 6 | **Top funded startups.** Flipkart ($4.76B), Paytm ($3.15B) and Ola ($2.05B) hold about 69% of the top-10 startup funding ($9.96B of $14.35B). Ola has the most deals (13), but Flipkart raised more in only 6 |
| 7 | **Data quality changed the story.** Correcting 11 amounts removed about $5.4B from the total. Without this step, Rapido Bike Taxi would have appeared as the single largest deal in the dataset |

---

## 7. Before vs After

| Check | Before | After |
|---|---|---|
| Shape | 3,044 × 10 | 3,044 × 16 |
| Total funding (`amount_usd`) | $38.14B | $32.76B |
| Mean deal size | ≈ $18.4M | ≈ $15.8M |
| Median deal size | $1.75M | $1.70M |
| Largest single deal | $3.9B (Rapido Bike Taxi) | $2.5B (Flipkart, Aug 2017) |
| Rows flagged as outliers | 118 | 110 |
| Wrong amounts corrected | 0 | 11 (original kept in `amount_original`) |
| Unique startup names | 2,457 | 2,242 |
| Unique city labels (incl. `Unknown`) | 77 | 67 |
| Missing `amount_usd` | 971 | 971 (not imputed) |
| Duplicate rows | 0 | 0 |

> **Why the mean fell more than the median:** the median is robust to extreme values, so removing eleven inflated amounts barely moves it ($1.75M to $1.70M). The mean is pulled up by large values, so it drops by about $2.6M. This is the reason the analysis reports medians next to totals.

---

## 8. Final Data Dictionary

| Column | Type | Description |
|---|---|---|
| `sr_no` | int | Original serial number |
| `date` | datetime | Date of the funding round |
| `year` | int | Year taken from `date` (new) |
| `startup_name` | text | Company name as in the cleaned source (unchanged) |
| `startup_clean` | text | Startup name with spelling variants merged (new) |
| `industry` | text | Industry vertical (`Unknown` if blank) |
| `subvertical` | text | More specific business description (`Not provided` if blank) |
| `city` | text | City as in the cleaned source (unchanged) |
| `city_clean` | text | City with typos, neighbourhoods and codes merged (new) |
| `investors` | text | Investor name(s) (`Undisclosed` if blank) |
| `investment_type` | text | Standardised round type (16 categories) |
| `amount_usd` | float | Funding amount in USD **after correction** (`NaN` if undisclosed) |
| `amount_original` | float | Funding amount before any correction (new) |
| `amount_fixed` | bool | `True` for the 11 rows whose amount was corrected (new) |
| `outlier_after_fix` | bool | `True` if the amount is above Q3 + 3 × IQR of its investment type, after corrections (new) |
| `remarks` | text | Extra notes (`Not provided` if blank) |

---

## 9. Decisions & Assumptions

- **Outliers are flagged, not deleted.** A row is changed only when public news reports confirm the correct amount. Large but real deals (Flipkart, Paytm, Udaan) stay in the data.
- **The IQR rule runs within each investment type,** with a multiplier of 3. A $50M seed round and a $50M Series C are not the same kind of event, so a flat cut-off would be unfair.
- **Every change is traceable.** The original amount is kept in `amount_original`, the corrected rows are marked in `amount_fixed`, and the original `startup_name` and `city` columns are kept next to their cleaned versions.
- **The median is reported next to the total** everywhere, because a few very large deals dominate the totals.
- **Amounts were not imputed.** An undisclosed amount is a fact about the deal, not an error to fill. Instead, the share of missing amounts is shown wherever totals are compared.
- **Minimum sample size:** industries and cities are ranked only if they have at least 10 deals with an amount. `Unknown` labels are excluded from those rankings.
- **Delhi NCR** is defined as New Delhi + Gurugram + Noida. This is an analysis choice, not a label in the data.
- **Each investor gets credit for every deal they joined,** counted once per deal. The ranking is by number of deals, not by amount invested.
- **Rounds beyond Series D are not compared,** and `Private Equity` is left out of the stage chart because it mixes very different deals.
- **`Ola Electric`, `OYOfit` and `Paytm Marketplace` are kept separate** from their parent brands because they are different companies or business units.
- **`True North` is excluded from the startup ranking** because it is a private equity fund raise, not startup funding.

---

## 10. Limitations

- **About 32% of `amount_usd` is missing** (971 rows), so all totals are understated, and 2016 is affected most (41% missing).
- **Smaller amount errors may remain.** Only the extreme cases flagged by the IQR rule and the manual review were checked. The 11 corrections rest on public news reports from 2018 and 2019.
- **2020 contains only January** (7 deals), so it cannot be compared with full years.
- **The fall in deal count after 2016 may partly reflect how the data was collected,** not only the market.
- **Industry labels are free text,** so similar labels such as `EdTech` and `Education` are not merged and about 800 labels remain.
- **Startup names with different words may remain split.** Spelling variants were merged for all startups, but only Ola and OYO were merged by hand.
- **Investor names are not fully standardised.** Only the clear variants were merged, and labels that join two named investors (for example `Premji Invest & Mayfield Fund`) were not split, so those investors are slightly undercounted.
- **Some `Private Equity` rows may be fund raises,** not startup funding (`True North` was excluded, others may remain).
- **Cities with two names or only a state** (for example `Karnataka`) were not assigned to a city.
- **Series E to J** have only 1 or 2 deals each, so their medians are not reliable.

---

## 11. How to Reproduce

1. Clone this repository.
2. Upload `cleaned_startup_funding.csv` (the Task 1 output) to Google Drive.
3. Open `Indian_Startup_Funding_Exploratory_Data_Analysis.ipynb` in Google Colab.
4. Update the file paths in the load cell (Step 1) and the export cell (Step 9) to match your Drive.
5. Run all cells (**Runtime → Run all**), in order, because later cells depend on columns created earlier (`outlier`, `to_verify`, `amount_original`, `city_clean`, `startup_clean`).
6. The final table is written as `startup_funding_eda.csv`.

**Requirements:** Python 3, `pandas`, `matplotlib`. The Drive mount and the download call are Colab-specific; outside Colab, replace them with local file paths.

---

## 12. Key Learnings

- **Check the data before trusting the totals.** One wrong value ($3.9B for a $54M deal) was enough to change the top of every ranking.
- **A single cut-off does not fit every group.** Applying the outlier rule within each investment type found errors that a flat $100M limit would have missed, and avoided flagging normal late-stage deals.
- **Flag first, decide second.** A statistical flag is a reason to look, not a reason to delete; the decision was made only after reading the news reports.
- **Keep the original next to the fix.** Adding `amount_original` and `amount_fixed` makes every correction reviewable.
- **Use the median for skewed money data,** and show how much of the data is missing next to every total.
- **Know when a sample is too small to compare.** Rounds with 1 or 2 deals do not have a reliable median, so they were described, not ranked.
- **Name variants hide real results.** Merging investor and startup spellings changed the rankings (Sequoia Capital 72 to 112 deals).
- **Every finding needs a caveat.** Writing the limitation next to the result keeps the conclusions honest.

---

## 13. Next Steps

- Build an **interactive dashboard** (Power BI, Tableau or another BI tool) on `startup_funding_eda.csv`, with filters for year, industry, city and investment type.
- Show the share of missing amounts on the dashboard, so that totals are not read as complete.
- Merge the remaining industry labels (for example `EdTech` and `Education`) if the dashboard needs fewer categories.

---

## 14. Author

**Sushant Kumar Yadav**
Aspiring Data Analyst · B.Com (Hons), University of Delhi

LinkedIn: [www.linkedin.com/in/sushantkumaryadav310899](https://www.linkedin.com/in/sushantkumaryadav310899/)
GitHub: [Skysushant7366](https://github.com/Skysushant7366)
Portfolio: [https://skysushant7366.github.io/Sushant-portfolio/](https://skysushant7366.github.io/Sushant-portfolio/)
*Completed as Task 2 of the Data Analyst Internship at SWYNEX Technologies.*
