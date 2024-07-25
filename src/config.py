from pathlib import Path

BASE_PATH = Path("../data")

METADATA_PATH = Path(BASE_PATH, "Metadata OneStopGaze L1 - metadata.csv")
QUESTIONNAIRE_PATH = Path(BASE_PATH, "questionnaire.json")

FULL_REPORT_PATH = Path(BASE_PATH, "full_report.csv")
SESSION_SUMMARY_PATH = Path(BASE_PATH, "session_summary.csv")
REPORTS_BASE_PATH = Path(BASE_PATH, "Outputs")
IA_P_PATH = Path(REPORTS_BASE_PATH, "IA reports/ia_P.tsv")
IA_Q_PATH = Path(REPORTS_BASE_PATH, "IA reports/ia_Q.tsv")
IA_A_PATH = Path(REPORTS_BASE_PATH, "IA reports/ia_A.tsv")
IA_T_PATH = Path(REPORTS_BASE_PATH, "IA reports/ia_T.tsv")
IA_Q_preview_PATH = Path(REPORTS_BASE_PATH, "IA reports/ia_Q_preview.tsv")
TRIAL_P_PATH = Path(REPORTS_BASE_PATH, "trial reports/trial_P.tsv")
TRIAL_QA_PATH = Path(REPORTS_BASE_PATH, "trial reports/trial_QA.tsv")
TRIAL_q_preview_PATH = Path(REPORTS_BASE_PATH, "trial reports/trial_Q_preview.tsv")
TRIAL_T_PATH = Path(REPORTS_BASE_PATH, "trial reports/trial_T.tsv")

# ENGLISH_COUNTRIES = [
#     "United States",
#     "Australia",
#     "United Kingdom",
#     "Canada",
#     "Ireland",
#     "South Africa",
#     "Nigeria",
# ]
