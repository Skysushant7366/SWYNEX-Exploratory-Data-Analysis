<div align="center">

# Indian Startup Funding: Exploratory Data Analysis

**SWYNEX Technologies · Data Analyst Internship · Task 2**

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Analysis-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisation-11557C)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-2EA44F)

*A cleaned startup-funding table audited for outliers, corrected where public evidence exists, and analysed by year, industry, city, investor, funding stage and startup.*

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

**Objective:** Use the cleaned dataset from Task 1 to understand Indian startup funding: how funding changed over time, which industries, cities, investors and funding stages matter most, and which startups attracted the largest rounds.

**Approach:** The analysis was done in a Google Colab notebook with `pandas` and `matplotlib`. Every section follows the same pattern: **check the data first, analyse second, and state the findings and caveats next to each result.** Nothing was deleted; every change is flagged, explained and reversible.

**Outcome:**
- **11 wrong funding amounts** (rupee values stored as USD, plus data-entry errors) were found through outlier detection, checked against public news reports and corrected. The 11 rows carried about **$5.46B before** the fix and about **$0.08B after** it, which lowers the dataset total by about **$5.38B (14%)**.
- **Six analyses** (year, industry, city, investor, funding stage, startup) with charts, findings and caveats written next to each result.
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
SWYNEX-Exploratory-Data-Analysis/
├── cleaned_startup_funding.csv                              # Input: Task 1 output
├── startup_funding_eda.csv                                  # Final analysis-ready data (16 columns)
├── Indian_Startup_Funding_Exploratory_Data_Analysis.ipynb   # Full notebook with every step
├── README.md                                                # This document
├── LICENSE                                                  # MIT license
└── images/                                                  # Charts used in this README
    ├── 01_yearly_trend.png
    ├── 02_top_industries.png
    ├── 03_top_cities.png
    ├── 04_top_investors.png
    ├── 05_median_by_stage.png
    └── 06_top_startups.png
```

---

## 4. Data Audit (Before Analysis)

Checked using `df.shape`, `df.info()`, `df.isna().sum()`, `(df == 'Unknown').sum()`, `df.duplicated().sum()` and `df.describe()`.

| # | Check | Result |
|---|---|---|
| 1 | Shape | 3,044 rows × 10 columns |
| 2 | Data types | `date` is `datetime64`, `amount_usd` is `float64`, `sr_no` is `int64`, the rest are text |
| 3 | Missing values | Only `amount_usd` has real missing values: **971 rows (31.9%)**, because the amount was never disclosed |
| 4 | `Unknown` placeholder labels | `industry` 171 · `city` 180 · `investment_type` 4 (placeholders from Task 1; kept, and excluded per analysis) |
| 5 | Empty strings | 0 in every text column |
| 6 | Duplicate rows | **0** (checked after dropping `sr_no`, which is unique by design) |
| 7 | Skewed amounts | Mean ≈ **$18.4M** against a median of **$1.75M**; maximum **$3.9B**. A few extreme values pull the average far above the typical deal |

**Why the last row matters most:** the gap between the mean and the median is the first sign that some amounts are wrong or extreme. It is what triggers the outlier work in Step 2.

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

---

### Step 2: Outlier detection and correction

This is the most important step of the project, because wrong amounts change every total that follows.

**2.1 Spot-check the largest values.** The 40 largest amounts were listed, and every deal above $100M was flagged as `suspect` (**56 rows**). Seed/Angel deals of $50M or more were listed separately (**5 rows**: Digital Mall Asia, FreshVnF, Shape, Rein Games, Dhruva Space), because a seed round that large is very unlikely. `suspect` is only a "look here" list: it also contains real large deals such as Flipkart and Udaan.

**2.2 Outliers within each investment type.** A normal deal size differs by round, so one flat cut-off is unfair: $50M is extreme for a seed round and ordinary for private equity. Amounts above `Q3 + 3 × IQR` were therefore flagged **within each investment type**. Using 3 instead of the usual 1.5 means only extreme values are flagged.

```python
def flag(g):
    q1, q3 = g.quantile(.25), g.quantile(.75)
    return g > q3 + 3*(q3 - q1)

df['outlier'] = df.groupby('investment_type').amount_usd.transform(flag)
```

**Result: 118 rows flagged.**

| Round | Flagged |
|---|---:|
| Private Equity | 82 |
| Seed/Angel | 28 |
| Series B | 3 |
| Other, Pre-Series A, Series A, Series C, Series D | 1 each |

A flag is a decision aid, not a verdict. Very small groups cannot be tested well by this rule.

**2.3 Manual review.** The flagged rows were reviewed by hand, round by round:

| Round | Flagged | Decision |
|---|---:|---|
| Seed/Angel | 28 | The top 5 ($50M to $220M) are far too large for a seed round. The other 23 ($2.4M to $12M) are high but possible, so they were left unchanged |
| Series B | 3 | Vogo Automotive ($283M) and Rapido Bike Taxi ($3.9B) look unrealistic. Automation Anywhere ($300M) is a large but possible deal |
| One-row groups | 5 | Paytm ($1B, Other) and Udaan ($585M, Series D) match news reports and were left unchanged. GOQii ($450M, Series C), Recykal ($26M, Pre-Series A) and Kuvera ($45M, Series A) were checked against 2019 news reports and marked for verification |

**2.4 Mark rows to verify.** Ten rows with unrealistic amounts were marked `to_verify`.

**2.5 Fix the wrong amounts.** Each of the 10 rows was checked against public news reports from 2018 and 2019. Most were **rupee amounts stored as USD** (for example, Rs 22 Cr shown as $220M); Recykal and Kuvera were data-entry errors. Furtados School of Music ($200M) was **not** flagged by the IQR rule because its investment type group is very small, so it was corrected as well (Rs 20 Cr, about $2.8M).

| Startup | Original amount | Corrected amount | Reason |
|---|---:|---:|---|
| Rapido Bike Taxi | $3,900,000,000 | $54,000,000 | Rs 390 Cr |
| Vogo Automotive | $283,000,000 | $4,000,000 | Rs 28.3 Cr |
| GOQii | $450,000,000 | $6,240,000 | Rs 45 Cr |
| Digital Mall Asia | $220,000,000 | $3,060,000 | Rs 22 Cr |
| Furtados School of Music | $200,000,000 | $2,800,000 | Rs 20 Cr |
| FreshVnF | $140,000,000 | $2,000,000 | $2M reported |
| Shape | $100,000,000 | $200,000 | $200K reported |
| Dhruva Space | $50,000,000 | $700,000 | Rs 5 Cr |
| Rein Games | $50,000,000 | $700,000 | Rs 5 Cr |
| Kuvera | $45,000,000 | $4,500,000 | $4.5M reported |
| Recykal | $26,000,000 | $2,000,000 | $2M reported |

Rupee amounts convert at roughly Rs 71 to 72 per USD, so the corrected values are approximate.

```python
# Keep the original amount (only on the first run)
if 'amount_original' not in df.columns:
    df['amount_original'] = df.amount_usd

m = df.to_verify
new = df.loc[m, 'startup_name'].map(fix)
assert new.notna().all(), 'a to_verify row has no fix'
df.loc[m, 'amount_usd'] = new
```
The `assert` stops the notebook if a marked row has no correction, so nothing is changed silently. The guard on `amount_original` makes the cell safe to re-run without overwriting the backup.

**2.6 Re-check after the fixes.** The same IQR rule was run again and saved in a new column `outlier_after_fix`, so the first flags stay visible.

| Check | Before fix | After fix |
|---|---:|---:|
| Rows flagged | 118 | **110** |
| Seed/Angel | 28 | 24 |
| Series B | 3 | 1 (Automation Anywhere) |
| Pre-Series A, Series C | 1 each | 0 |
| Private Equity | 82 | 82 |

One row is new: **Engineer.ai** ($29.5M, Series A). It is only above the limit because the Series A threshold fell after Kuvera was corrected, so it is left unchanged. Private Equity (82) is a mixed group that includes real large deals such as Flipkart and Paytm, so those rows are left unchanged too.

---

### Step 3: Year-wise trend

Deals were grouped by year, comparing the number of deals, the total funding and the **median** deal size. The median is used because a few very large deals pull the total up. The number of deals with an amount is counted too, because many amounts are missing.

```python
df['year'] = df.date.dt.year
yr = df.groupby('year').agg(deals=('sr_no','count'),
                            with_amount=('amount_usd','count'),
                            total=('amount_usd','sum'),
                            median=('amount_usd','median'))
yr['missing_pct'] = (1 - yr.with_amount/yr.deals) * 100
```

| Year | Deals | With amount | Missing | Total | Median |
|---|---:|---:|---:|---:|---:|
| 2015 | 936 | 654 | 30% | $8.67B | $1.50M |
| 2016 | 993 | 586 | 41% | $3.83B | $1.00M |
| 2017 | 687 | 456 | 34% | $10.43B | $2.25M |
| 2018 | 310 | 265 | 15% | $5.02B | $3.80M |
| 2019 | 111 | 105 | 5% | $4.42B | $6.24M |
| 2020 | 7 | 7 | 0% | $0.39B | $9.00M |

![Yearly trend](images/01_yearly_trend.png)

**Findings**
- The number of deals falls every year after 2016 (993 to 111 in 2019), but the median deal size keeps rising ($1.0M in 2016 to $6.2M in 2019). Fewer, larger deals are what the data shows.
- Total funding does not follow the number of deals: it peaks in 2017 ($10.4B) because of a few very large deals, then falls to about $5B in 2018 and 2019.

**Caveats**
- 2020 contains only January (7 deals), so it cannot be compared with the full years.
- The share of deals with a missing amount falls from 41% in 2016 to 5% in 2019, so the earlier totals are understated more than the later ones.
- The drop in the number of deals may partly reflect how the data was collected, not only the market.

---

### Step 4: Industry-wise funding

Deals were grouped by industry. Only industries with **at least 10 deals that have an amount** are ranked, so the median is reliable and one-off labels do not distort the list. Deals with an `Unknown` industry (171 rows) are excluded.

| Industry | Deals | With amount | Total | Median |
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

![Top industries](images/02_top_industries.png)

**Findings**
- eCommerce ($8.0B) and Consumer Internet ($6.3B) together hold about **68%** of the top-10 industry funding.
- Consumer Internet has the most deals (941) but one of the lowest medians ($1.0M): many small deals. eCommerce has a higher median ($3.0M).
- FinTech (median $10M), Finance ($6.0M) and Logistics ($6.0M) have fewer but larger deals.

**Caveats**
- Industry labels are free text, so similar labels (for example `EdTech` and `Education`) stay separate.
- About a third of the deals in the largest industries have no amount (31% in eCommerce, 37% in Consumer Internet), so the totals are understated.

---

### Step 5: City-wise funding

**5.1 Merge city labels.** The 30 most common labels were mostly clean, but a few were misspelled or were neighbourhoods of a larger city (`Kormangala`, `Andheri`, `Taramani`), and some were not Indian cities or held two cities (`Bengaluru and Gurugram`). Only clear typos, neighbourhoods and codes were mapped into a new column `city_clean` (**13 rows changed**); the original `city` column is untouched. Labels with two cities or only a state were left as they are, because the intended city cannot be known.

```python
city_fix = {'Kormangala':'Bengaluru', 'Andheri':'Mumbai', 'Chembur':'Mumbai',
            'Taramani':'Chennai', 'Nw Delhi':'New Delhi', 'Kolkatta':'Kolkata',
            'Ahemdabad':'Ahmedabad', 'Bhubneswar':'Bhubaneswar',
            'San Jose,':'San Jose', 'SFO':'San Francisco', 'US':'USA'}
df['city_clean'] = df.city.replace(city_fix)
```

**5.2 Funding by city.** Cities with at least 10 deals that have an amount are ranked. Deals with an `Unknown` city (180 rows) are excluded.

| City | Deals | With amount | Total | Median |
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

**5.3 Delhi NCR.** `Delhi NCR` is not a label in the data, so New Delhi, Gurugram and Noida were added together (other nearby cities such as Faridabad are not included).

| Region | Deals | With amount | Total |
|---|---:|---:|---:|
| Delhi NCR (New Delhi + Gurugram + Noida) | **899** | 571 | $8.26B |
| Bengaluru | 851 | 593 | $14.60B |

![Top cities](images/03_top_cities.png)

**Findings**
- Bengaluru leads with about $14.6B, roughly half of the top-10 city total ($29.7B). Mumbai ($4.8B) is second among single cities.
- Delhi NCR raised about $8.3B, so it is the second-largest hub. It has **more deals than Bengaluru (899 vs 851) but much less money**.
- The median deal size is between $1.0M and $2.5M in all ten cities, so Bengaluru's lead in total comes mostly from a few very large deals, not from a higher typical deal.

**Caveats**
- NCR is the sum of three city labels; the chart shows these three cities separately.
- About a third of the deals have no amount (30% in Bengaluru, 36% in Delhi NCR), so the totals are understated.

---

### Step 6: Top investors

Each row can list several investors separated by commas. They were split so that every investor gets credit for every deal they joined, and investors are ranked by the **number of deals** (not by money invested).

**6.1 Clean the investor names.** The raw labels contained variants of the same firm (`Sequoia Capital`, `Sequoia India`, `Sequoia Capital India`) and placeholders (`Undisclosed`, `Undisclosed Investors`, `Undisclosed investors`). The cleaning rules were:
- generic endings such as `& Others` or `and existing investors` are removed, so the investor name stays;
- clear name variants are merged (for example `Sequoia India` and `Sequoia Capital India` into `Sequoia Capital`);
- generic labels such as `Undisclosed` are removed;
- an investor is counted once per deal.

Labels that join two named investors (for example `Premji Invest & Mayfield Fund`) are not split, because some real names also contain `&` or `and`.

```python
inv = df[['sr_no','investors']].copy()
inv['investor'] = inv.investors.str.split(',')
inv = inv.explode('investor')
inv['investor'] = inv.investor.str.strip()

# remove generic endings such as "& Others" or "and existing investors"
tail = r'\s*(?:&|and)\s+(?:others?\b|existing\b|one more\b).*$'
inv['investor'] = inv.investor.str.replace(tail, '', regex=True, case=False).str.strip()
# ...then merge name variants with a mapping (inv_fix) and drop generic labels
```

Merging the variants changes the ranking: **Sequoia Capital rose from 72 to 112 deals and Tiger Global from 28 to 52**.

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

![Top investors](images/04_top_investors.png)

**Findings**
- Sequoia Capital (112 deals) and Accel Partners (84) are the most active investors, well ahead of Tiger Global (52).
- Individual angels such as Ratan Tata (31 deals) rank alongside venture funds, and Indian Angel Network (43) is a group of angels.

**Caveats**
- The ranking is by number of deals, not by the amount invested.
- `Undisclosed` and other generic labels were removed and the clear name variants were merged; other variants may remain.
- Labels that join two named investors were not split, so those investors are slightly undercounted.

---

### Step 7: Funding by investment type

Deals were grouped by investment type (the 4 `Unknown` rows are excluded).

| Investment type | Deals | With amount | Total | Median |
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
| Series E to J | 1 to 2 each | | | not reliable |

**Series J check.** The only Series J row is Ola Cabs ($1M, July 2019). Ola's Series J was a large round raised in several parts during 2019, so this row is only a small part of it. A single row cannot give a reliable median, so rounds beyond Series D are not compared. The amount was left unchanged because it does not match a reported amount for July 2019.

The chart shows only Seed/Angel and Series A to D, where there are enough deals. `Private Equity` is left out because it is a mixed group.

![Median deal size by stage](images/05_median_by_stage.png)

**Findings**
- Private Equity holds about **84%** of the $32.76B total, while Seed/Angel has the most deals (1,542) but only about **1.5%** of the money.
- The median deal size rises with the round: $0.36M (Seed/Angel), $4.0M (Series A), $15.1M (Series B), $44.0M (Series C) and $65.0M (Series D).

**Caveats**
- `Private Equity` mixes very different deals, so its median is not comparable with the numbered rounds, and some rows may be fund raises, not startup funding.
- The `Other` group ($1.2B) is mostly one deal, Paytm's $1B round.
- Series E to J have only 1 or 2 deals each, so their medians are not reliable.

---

### Step 8: Top startups by funding

The same startup can appear under different spellings (for example `Ola` and `Ola Cabs`), which would split its total across several rows.

**8.1 Merge name variants.** Names that differ only by capital letters, spaces, punctuation or endings such as `.com`, `Pvt` and `Ltd` were matched with a normalised key. **189 startups** appeared under more than one spelling (for example `ZoomCar` and `Zoomcar`, or `AYE Finance` and `AyeFinance`). All spellings were mapped to the most common one in a new column `startup_clean`; the original `startup_name` is unchanged. This reduced the unique names from **2,457 to 2,244**.

```python
def name_key(s):
    s = re.sub(r'\.com|\.in|\bpvt\b|\bltd\b|\bprivate\b|\blimited\b', '', s.lower())
    return re.sub(r'[^a-z0-9]', '', s)

names = df[['startup_name']].copy()
names['key'] = names.startup_name.map(name_key)
canon = names.groupby('key').startup_name.agg(lambda x: x.value_counts().idxmax())
df['startup_clean'] = names.key.map(canon)
```

**8.2 Merge names that use different words.** `Ola Cabs`, `Ola`, `OYO Rooms` and `Oyo` use different words, so they were merged by hand. `Ola Electric`, `OYOfit` and `Paytm Marketplace` are different companies or business units and stay separate.

```python
brand_fix = {'Ola Cabs': 'Ola', 'OYO Rooms': 'OYO', 'Oyo': 'OYO'}
df['startup_clean'] = df.startup_clean.replace(brand_fix)
```

**8.3 Rank by total funding.** `True North` ($600M) is excluded: it is a private equity fund raise (a fund closing), not funding received by a startup.

| Rank | Startup | Deals | With amount | Total |
|---:|---|---:|---:|---:|
| 1 | Flipkart | 6 | 6 | $4.76B |
| 2 | Paytm | 7 | 5 | $3.15B |
| 3 | Ola | 13 | 12 | $2.05B |
| 4 | OYO | 9 | 9 | $1.00B |
| 5 | Udaan | 4 | 4 | $0.87B |
| 6 | Snapdeal | 2 | 2 | $0.70B |
| 7 | Byju's | 7 | 6 | $0.53B |
| 8 | BigBasket | 5 | 5 | $0.51B |
| 9 | Zomato | 4 | 4 | $0.44B |
| 10 | PolicyBazaar | 4 | 4 | $0.34B |

![Top startups](images/06_top_startups.png)

**Findings**
- Flipkart ($4.76B), Paytm ($3.15B) and Ola ($2.05B) are the top three. Together they hold about **69%** of the top-10 funding ($9.96B of $14.35B).
- Ola has the most deals (13), but Flipkart raised more money in only 6 deals, so its rounds were much larger.
- OYO ($1.0B) is fourth. Udaan, Snapdeal, Byju's, BigBasket, Zomato and PolicyBazaar are each between $0.34B and $0.87B.

**Caveats**
- Paytm's total includes a $1B row from November 2019 where the only investor listed is Vijay Shekhar Sharma and the type is `Other`. Without it, Paytm would be $2.15B, still third.
- `Paytm Marketplace`, `Ola Electric` and `OYOfit` are kept separate from their parent brands.
- Rows with a missing amount are not counted (for example, Paytm has 7 deals but 5 with an amount).

---

### Step 9: Export
```python
final_cols = ['sr_no', 'date', 'year', 'startup_name', 'startup_clean', 'industry',
              'subvertical', 'city', 'city_clean', 'investors', 'investment_type',
              'amount_usd', 'amount_original', 'amount_fixed', 'outlier_after_fix', 'remarks']
out = df[final_cols]
out.to_csv('/content/drive/MyDrive/startup_funding_eda.csv', index=False)
print(out.shape)   # (3044, 16)
```
Working columns used only to hunt for errors (`suspect`, `to_verify`, `outlier`) are left out so the final file stays clean.

---

## 6. Key Findings

| # | Finding |
|---|---|
| 1 | **Fewer, bigger deals.** Deals fell from 993 (2016) to 111 (2019), while the median deal size rose from $1.0M to $6.2M. Total funding peaked in 2017 ($10.4B) because of a few very large deals. |
| 2 | **Funding is concentrated by industry.** eCommerce ($8.0B) and Consumer Internet ($6.3B) hold about 68% of the top-10 industry funding. |
| 3 | **Bengaluru leads by money, Delhi NCR by deal count.** Bengaluru raised about $14.6B, roughly half of the top-10 city total. Delhi NCR (New Delhi, Gurugram and Noida together) has more deals (899 vs 851) but raised about $8.3B. |
| 4 | **Stage drives deal size.** The median rises from $0.36M (Seed/Angel) to $4.0M (Series A), $15.1M (Series B), $44.0M (Series C) and $65.0M (Series D). Private Equity holds about 84% of the total funding. |
| 5 | **Most active investors.** Sequoia Capital (112 deals) and Accel Partners (84) lead, well ahead of Tiger Global (52). Name cleaning raised Sequoia from 72 to 112 deals. |
| 6 | **Top funded startups.** Flipkart ($4.76B), Paytm ($3.15B) and Ola ($2.05B) hold about 69% of the top-10 startup funding. |
| 7 | **Data quality changed the story.** Correcting 11 amounts removed about $5.4B from the total, and without the fix the largest deal in the data would have been a $3.9B Rapido round that never happened at that size. |

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
| Wrong amounts corrected | 0 | 11 |
| Unique startup names | 2,457 | 2,244 (`startup_clean`) |
| Investor ranking leader | Sequoia Capital, 72 deals | Sequoia Capital, 112 deals |

The mean moves far more than the median, which is why the median is reported next to every total.

---

## 8. Final Data Dictionary

| Column | Type | Description |
|---|---|---|
| `sr_no` | int | Original serial number |
| `date` | datetime | Date of the funding round |
| `year` | int | Year taken from `date` |
| `startup_name` | text | Company name as in the cleaned source |
| `startup_clean` | text | Startup name with spelling variants merged |
| `industry` | text | Industry vertical (`Unknown` if blank) |
| `subvertical` | text | More specific business description |
| `city` | text | City as in the cleaned source |
| `city_clean` | text | City with clear typos and neighbourhoods merged |
| `investors` | text | Investor name(s) as in the cleaned source |
| `investment_type` | text | Standardised round type |
| `amount_usd` | float | Funding amount in USD after correction (`NaN` if undisclosed) |
| `amount_original` | float | Funding amount before correction |
| `amount_fixed` | bool | `True` if the amount was corrected (11 rows) |
| `outlier_after_fix` | bool | `True` if the amount is still above the IQR limit after correction (110 rows) |
| `remarks` | text | Additional notes |

---

## 9. Decisions & Assumptions

- **Outliers are flagged, not deleted.** A row is changed only when public news reports confirm the correct amount.
- **The IQR rule runs within each investment type**, with a multiplier of 3 so that only extreme values are flagged.
- **The original is kept next to the fix** (`amount_original`, `amount_fixed`), so every correction can be audited or reversed.
- **The median is reported next to the total**, because a few very large deals dominate totals.
- **Amounts were not imputed.** An undisclosed amount is a real fact, not a data error; filling it would invent money that was never raised.
- **Minimum sample size:** industries and cities are ranked only if they have at least 10 deals with an amount.
- **Delhi NCR** is defined as New Delhi + Gurugram + Noida only.
- **Each investor gets credit for every deal they joined**, counted once per deal.
- **True North is excluded from the startup ranking** because it is a fund raise, not startup funding. It stays in the dataset.
- **City and name merges were conservative:** only clear typos, neighbourhoods and spelling variants were merged. Ambiguous labels were left as they are.

---

## 10. Limitations

- About 32% of `amount_usd` is missing (971 rows), so totals are understated, most of all for 2016 (41% missing).
- Rupee-to-USD corrections rest on one or two news reports per row and use an approximate rate of Rs 71 to 72 per USD. The Shape correction is the least certain.
- Smaller cases of the same rupee-as-USD error may remain, because the IQR rule flags only extreme values.
- 2020 contains only January (7 deals), so it is not comparable to full years.
- The fall in the number of deals after 2016 may partly reflect how the data was collected, not only the market.
- Industry labels are free text and are not fully standardised (for example `EdTech` and `Education`).
- Some Private Equity rows may be fund raises, not startup funding. Only `True North` was verified and excluded.
- Startup names that use different words may remain split (only Ola and OYO were merged by hand), and investor names that join two investors in one label were not split.
- Series E to J have only 1 or 2 deals each, so their medians are not reliable.

---

## 11. How to Reproduce

1. Clone this repository.
2. Upload `cleaned_startup_funding.csv` (the Task 1 output) to Google Drive.
3. Open `Indian_Startup_Funding_Exploratory_Data_Analysis.ipynb` in Google Colab.
4. Update the file paths in the load cell and the export cell.
5. Run all cells **in order** (**Runtime → Run all**). The load cell rebuilds `df` from the original file, so re-running it later removes every correction made below it; run the notebook from the top again after that.
6. The final table is written as `startup_funding_eda.csv`.

**Requirements:** Python 3, `pandas`, `matplotlib`.

---

## 12. Key Learnings

- **Check the data before trusting the totals.** One wrong amount ($3.9B) distorted the mean, the yearly total and every ranking.
- **A single cut-off does not fit every group.** The IQR rule within each round type separated a suspicious seed deal from a normal private equity deal.
- **Flag first, decide second.** Flagging a row costs nothing; changing it needs evidence.
- **Keep the original next to the fix**, so the work can be audited.
- **Use the median for skewed money data**, and show how much data is missing next to every total.
- **Name variants hide real results.** Merging investor and startup spellings changed the rankings (Sequoia 72 to 112 deals).
- **Count what you rank.** Comparing deals with an amount against all deals gives opposite answers (Bengaluru 593 vs Delhi NCR 571, but 851 vs 899 in total), so each comparison states which one it uses.

---

## 13. Next Steps

- Build an interactive dashboard on `startup_funding_eda.csv`.
- Add missing-amount indicators to charts so totals are not misread as complete.
- Merge remaining industry labels such as `EdTech` and `Education` for cleaner segmentation.
- Check the remaining Private Equity rows for fund raises that are not startup funding.

---

## 14. Author

**Sushant Kumar Yadav**
Aspiring Data Analyst · B.Com (Hons), University of Delhi

LinkedIn: [www.linkedin.com/in/sushantkumaryadav310899](https://www.linkedin.com/in/sushantkumaryadav310899/)
GitHub: [Skysushant7366](https://github.com/Skysushant7366)
Portfolio: [https://skysushant7366.github.io/Sushant-portfolio/](https://skysushant7366.github.io/Sushant-portfolio/)
*Completed as Task 2 of the Data Analyst Internship at SWYNEX Technologies.*
