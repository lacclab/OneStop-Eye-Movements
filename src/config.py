from pathlib import Path

BASE_PATH = Path("../data")

METADATA_PATH = Path(BASE_PATH, "Metadata OneStopGaze L1 - metadata.csv")
FULL_REPORT_PATH = Path(BASE_PATH, "full_report.csv")
#ALL_DAT_FILES = Path(BASE_PATH, "all_dat_files_merged.tsv")

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
#TRIAL_P_PATH = Path(REPORTS_BASE_PATH, "trial reports/trial_P.tsv")
#TRIAL_QA_PATH = Path(REPORTS_BASE_PATH, "trial reports/trial_QA.tsv")
#TRIAL_Q_preview_PATH = Path(REPORTS_BASE_PATH, "trial reports/trial_Q_preview.tsv")
#TRIAL_T_PATH = Path(REPORTS_BASE_PATH, "trial reports/trial_T.tsv")

ENGLISH_SPEAKING_COUNTRIES = [
    "United States",
    "Australia",
    "United Kingdom",
    "Canada",
    "Ireland",
    "South Africa",
    "Nigeria",
]
