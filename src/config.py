from pathlib import Path

BASE_PATH = Path("../data")

METADATA_PATH = Path(BASE_PATH, "Metadata OneStopGaze L1 - metadata.csv")
FULL_REPORT_PATH = Path(BASE_PATH, "full_report.csv")

QUESTIONNAIRE_PATH = Path(BASE_PATH, "questionnaire.json")
SESSION_SUMMARY_PATH = Path(BASE_PATH, "session_summary.csv")

REPORTS_BASE_PATH = Path(BASE_PATH, "Outputs")
IA_P_PATH = Path(REPORTS_BASE_PATH, "raw_ia_reports/ia_P.tsv")
IA_Q_PATH = Path(REPORTS_BASE_PATH, "raw_ia_reports/ia_Q.tsv")
IA_A_PATH = Path(REPORTS_BASE_PATH, "raw_ia_reports/ia_A.tsv")
IA_T_PATH = Path(REPORTS_BASE_PATH, "raw_ia_reports/ia_T.tsv")
IA_QA_PATH = Path(REPORTS_BASE_PATH, "raw_ia_reports/ia_QA.tsv")
IA_Q_preview_PATH = Path(REPORTS_BASE_PATH, "raw_ia_reports/ia_Q_preview.tsv")
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
