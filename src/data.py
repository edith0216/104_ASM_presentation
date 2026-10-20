"""
Acquire, transform and validate the AI model concentration datasets.

Data sources:
1. Epoch AI Frontier Models dataset
2. Stanford AI Index 2025 Figure 1.3.6
3. Stanford AI Index 2025 Figure 1.3.7

Research question:
Is influential AI model development concentrated in fewer organisations?

Pipeline:
1. load_data(): load the raw Epoch AI Frontier Models dataset
2. clean_organisations(): harmonise and split organisation names
3. prepare_data(): parse years and calculate fractional weights
4. build_year_org_data(): aggregate annual organisation contributions
5. calculate_concentration(): calculate organisation count, CR3 and HHI
6. validate(): check processed data
7. save_outputs(): write dashboard-ready processed datasets

The pipeline produces:
1. frontier_model_org.csv
2. frontier_by_year_org.csv
3. frontier_concentration_by_year.csv
4. stanford_notable_2024.csv
5. stanford_notable_2014_2024.csv

Presentation and visualisation logic belongs in app.py or a dedicated charts module.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# 1. Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Data"
PROCESSED_DIR = DATA_DIR / "processed"
FRONTIER_FILE = DATA_DIR / "frontier_ai_models.csv"
STANFORD_2024_FILE = (DATA_DIR / "Stanford AI Index Figure 1.3.6.csv")
STANFORD_TOTAL_FILE = (DATA_DIR / "Stanford AI Index Figure 1.3.7.csv")

# Core analysis period used to align with the Stanford AI Index
START_YEAR = 2014
END_YEAR = 2024

# Load and clean Stanford AI Index data
def load_stanford_data(file_path, value_column, output_column):
    """Load and clean an organisation-level Stanford AI Index dataset."""

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    required_columns = ["Organization", value_column]

    missing = [
        col for col in required_columns
        if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean organisation names
    df["Organization"] = (df["Organization"]
                          .astype(str)
                          .str.strip())

    # Convert counts to numeric
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")

    # Remove rows with missing organisation or count
    df = df.dropna(subset=["Organization", value_column])

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Standardise output column names
    df = df.rename(columns={"Organization": "Organisation", value_column: output_column})

    return df


# Organization Name
# E.g.,DeepMind / Google Brain / Google Research / Google DeepMind are grouped under "Google".
ORG_MAPPING = {
    # Google-related organisations
    "Google": "Google",
    "Google Research": "Google",
    "Google Brain": "Google",
    "Google DeepMind": "Google",
    "DeepMind": "Google",

    # Meta-related organisations
    "Meta": "Meta",
    "Meta AI": "Meta",
    "Facebook": "Meta",
    "Facebook AI": "Meta",

    # Microsoft
    "Microsoft Research": "Microsoft",

    # Zhipu
    "Z.ai (Zhipu AI)": "Zhipu AI",
    "Zhipu AI": "Zhipu AI",

    # Universities
    "Carnegie Mellon University (CMU)": "Carnegie Mellon University",
    "Massachusetts Institute of Technology (MIT)": "MIT",
    "New York University (NYU)": "New York University",
    "University College London (UCL)": "University College London"}


# Load raw data
def load_data():
    """Load the raw Epoch AI Frontier Models dataset."""
    if not FRONTIER_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {FRONTIER_FILE}")

    df = pd.read_csv(FRONTIER_FILE)
    required_columns = ["Model", "Organization", "Publication date"]

    missing = [
        col for col in required_columns
        if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


# Clean organisation names
def clean_organisations(org_string):
    """Split, harmonise and deduplicate organisation names."""
    if pd.isna(org_string):
        return []

    orgs = [
        org.strip()
        for org in str(org_string).split(",")
        if org.strip()]

    orgs = [
        ORG_MAPPING.get(org, org)
        for org in orgs]

    # Remove duplicates after harmonisation
    orgs = list(dict.fromkeys(orgs))

    return orgs


# Prepare model-level data
def prepare_data(df):
    """Parse publication year and calculate fractional weights."""
    df = df.copy()

    # Parse publication date
    df["Publication date"] = pd.to_datetime(df["Publication date"], errors="coerce")
    df["Year"] = (df["Publication date"].dt.year.astype("Int64"))

    # Clean organisation names
    df["Org_list"] = (df["Organization"].apply(clean_organisations))

    # Number of organisations per model
    df["N_orgs"] = (df["Org_list"].apply(len))

    # Fractional attribution: 1 / N
    df["Fractional_weight"] = np.where(df["N_orgs"] > 0, 1 / df["N_orgs"], np.nan)

    return df


# Expand to model-organisation level
def expand_model_org(df):
    """Convert model-level records into model-organisation records."""
    expanded = df.explode("Org_list").copy()
    expanded = expanded.rename(columns={"Org_list": "Organisation"})
    expanded = expanded[expanded["Organisation"].notna()].copy()
    expanded["Organisation"] = (expanded["Organisation"].astype(str).str.strip())
    expanded = expanded[expanded["Organisation"] != ""].copy()
    expanded = expanded[expanded["Year"].between(START_YEAR, END_YEAR)].copy()

    return expanded


# Build annual organisation data
def build_year_org_data(expanded):
    """Aggregate annual fractional contributions by organisation."""
    year_org = (expanded.groupby(["Year", "Organisation"], as_index=False)
                .agg(Fractional_contribution=("Fractional_weight", "sum"),
                     Model_count=("Model","nunique")))

    # Total annual contribution
    year_org["Year_total"] = (year_org.groupby("Year")["Fractional_contribution"].transform("sum"))

    # Organisation share
    year_org["Share"] = (year_org["Fractional_contribution"] / year_org["Year_total"])
    year_org["Share_percent"] = (year_org["Share"] * 100)

    return year_org


# Calculate concentration metrics
def calculate_concentration(df, expanded, year_org):
    """Calculate annual frontier model count, organisation count, CR3 and HHI."""
    # Frontier model count
    model_count = (df[df["Year"].between(START_YEAR, END_YEAR)]
                   .groupby("Year")["Model"]
                   .nunique()
                   .reset_index(name="Frontier_model_count"))

    # Organisation count
    org_count = (expanded
                 .groupby("Year")["Organisation"]
                 .nunique()
                 .reset_index(name="Organisation_count"))

    # CR3
    ranked = year_org.sort_values(["Year", "Share"], ascending=[True, False])
    cr3 = (ranked
           .groupby("Year", group_keys=False)
           .head(3)
           .groupby("Year")["Share"]
           .sum()
           .reset_index(name="CR3"))

    cr3["CR3_percent"] = (cr3["CR3"] * 100)

    # HHI
    hhi = (year_org
           .assign(Share_squared=year_org["Share"] ** 2)
           .groupby("Year")["Share_squared"]
           .sum()
           .reset_index(name="HHI"))

    # Combine
    concentration = (model_count
                     .merge( org_count, on="Year", how="left")
                     .merge(cr3, on="Year", how="left")
                     .merge(hhi, on="Year", how="left")
                     .sort_values("Year")
                     .reset_index(drop=True))

    return concentration


# Validate processed data
def validate(df, expanded, year_org, concentration):
    """Check the main processing and concentration calculations. """

    print("\n" + "=" * 60)
    print("DATA PROCESSING CHECKS")
    print("=" * 60)

    print("\nOriginal frontier model records:", len(df))

    print("Missing organisation values:", df["Organization"].isna().sum())
    print("Models with no valid organisation:", (df["N_orgs"] == 0).sum())

    print("Multi-organisation models:", (df["N_orgs"] > 1).sum())
    print("Expanded model-organisation rows:", len(expanded))
    print("\nAnalysis period:", f"{START_YEAR}-{END_YEAR}")
    print("Number of year-organisation rows:", len(year_org))

    # Check annual shares sum to 1
    share_check = (year_org.groupby("Year")["Share"].sum())
    assert np.allclose(share_check.values, 1.0), "Annual organisation shares do not sum to 1."

    # Check CR3 range
    assert concentration["CR3"].between(0, 1).all(), "CR3 must be between 0 and 1."

    # Check HHI range
    assert concentration["HHI"].between(0, 1).all(), "HHI must be between 0 and 1."

    print("\nConcentration metrics:")
    print(concentration[
        ["Year", "Frontier_model_count",
         "Organisation_count", "CR3_percent", "HHI"]]
         .round({"CR3_percent": 2, "HHI": 3}).to_string(index=False))

    print("\nAll validation checks passed.")
    print("=" * 60)


# Validate Stanford AI Index data
def validate_stanford(stanford_2024, stanford_total):
    """Check Stanford organisation-level summary datasets."""
    checks = [
        (stanford_2024, "Notable_model_count_2024", "Stanford 2024"),
        (stanford_total, "Notable_model_count_2014_2024", "Stanford 2014-2024")]

    print("\n" + "=" * 60)
    print("STANFORD DATA CHECKS")
    print("=" * 60)

    for df, value_column, label in checks:
        assert not df.empty, f"{label} dataset is empty."
        assert df["Organisation"].notna().all(), f"{label} contains missing organisations."
        assert df[value_column].notna().all(), f"{label} contains missing model counts."
        assert (df[value_column] >= 0).all(), f"{label} contains negative model counts."
        assert not df["Organisation"].duplicated().any(), f"{label} contains duplicate organisations."
        print(f"{label}: {len(df)} organisations; total notable models = {df[value_column].sum():.0f}")

    print("Stanford validation checks passed.")
    print("=" * 60)


# Save processed datasets
def save_outputs(expanded, year_org, concentration, stanford_2024, stanford_total):
    """Save dashboard-ready processed datasets."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Model-organisation level
    model_org = expanded[
        ["Model", "Publication date",
         "Year", "Organization",
         "Organisation", "N_orgs",
         "Fractional_weight"]].copy()

    model_org.to_csv(PROCESSED_DIR / "frontier_model_org.csv", index=False)

    # Annual organisation level
    year_org.to_csv(PROCESSED_DIR / "frontier_by_year_org.csv", index=False)

    # Annual concentration metrics
    concentration.to_csv(PROCESSED_DIR / "frontier_concentration_by_year.csv", index=False)

    # Stanford AI Index summaries
    stanford_2024.to_csv(PROCESSED_DIR / "stanford_notable_2024.csv", index=False)
    stanford_total.to_csv(PROCESSED_DIR / "stanford_notable_2014_2024.csv", index=False)

    print("\nProcessed datasets saved to:")
    print(PROCESSED_DIR)

    print("- frontier_model_org.csv")
    print("- frontier_by_year_org.csv")
    print("- frontier_concentration_by_year.csv")
    print("- stanford_notable_2024.csv")
    print("- stanford_notable_2014_2024.csv")


def main():
    print("Loading Epoch AI Frontier Models dataset.")
    df = load_data()
    print(f"Loaded {len(df)} frontier model records.")

    print("Loading Stanford AI Index datasets...")
    stanford_2024 = load_stanford_data(
        STANFORD_2024_FILE,
        "Number of notable AI model in 2024",
        "Notable_model_count_2024")

    stanford_total = load_stanford_data(
        STANFORD_TOTAL_FILE,
        "Number of notable AI model, 2014-24",
        "Notable_model_count_2014_2024")

    df = prepare_data(df)
    expanded = expand_model_org(df)
    year_org = build_year_org_data(expanded)
    concentration = calculate_concentration(df, expanded, year_org)

    validate(df, expanded, year_org, concentration)
    validate_stanford(stanford_2024, stanford_total)

    save_outputs(expanded, year_org, concentration, stanford_2024, stanford_total)

    print("\nData processing completed successfully.")


if __name__ == "__main__":
    main()