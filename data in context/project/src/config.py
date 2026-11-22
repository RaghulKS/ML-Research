"""Central configuration for the MIMIC-IV cohort analysis pipeline."""
from pathlib import Path

# Resolve paths relative to the project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent  # workspace root where the raw CSV lives

# Directories
DIRS = {
    "data": PROJECT_ROOT / "data",
    "figures": PROJECT_ROOT / "figures",
    "tables": PROJECT_ROOT / "tables",
    "report": PROJECT_ROOT / "report",
    "notebooks": PROJECT_ROOT / "notebooks",
}

# Input / output files
RAW_CSV = DATA_ROOT / "bq-results-20260726-043100-1785041031640.csv"
CLEANED_CSV = DIRS["data"] / "cleaned_dataset.csv"

# Figure settings
FIG_DPI = 300
FIG_FORMAT = "png"
plt_rc = {
    "figure.dpi": 120,
    "savefig.dpi": FIG_DPI,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.autolayout": False,
}

# Disease flag columns
DISEASE_FLAGS = ["has_diabetes", "has_kidney_disease", "has_both"]

# Numeric columns of interest
NUMERIC_COLS = [
    "anchor_age",
    "hospital_length_of_stay_days",
    "icu_length_of_stay_days",
    "hospital_expire_flag",
    "icu_admission",
]

# Datetime columns
DATETIME_COLS = ["admittime", "dischtime", "intime", "outtime"]

# Categorical columns
CATEGORICAL_COLS = [
    "gender",
    "anchor_year_group",
    "admission_type",
    "admission_location",
    "discharge_location",
    "insurance",
    "language",
    "marital_status",
    "race",
    "icd_version",
    "first_careunit",
    "last_careunit",
]


def ensure_dirs():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
