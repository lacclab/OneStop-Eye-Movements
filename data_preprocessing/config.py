from pathlib import Path

BASE_PATH = Path("../metadata")

METADATA_PATH = Path(BASE_PATH, "Metadata OneStopGaze L1 - metadata.csv")
FULL_REPORT_PATH = Path(BASE_PATH, "full_report.csv")

QUESTIONNAIRE_PATH = Path(BASE_PATH, "questionnaire.json")
SESSION_SUMMARY_PATH = Path(BASE_PATH, "session_summary.csv")

REPORTS_BASE_PATH = Path(BASE_PATH, "Outputs")
TRIAL_PATH = Path(REPORTS_BASE_PATH, "trial_report.tsv")


ENGLISH_SPEAKING_COUNTRIES = [
    "United States",
    "Australia",
    "United Kingdom",
    "Canada",
    "Ireland",
    "South Africa",
    "Nigeria",
]
