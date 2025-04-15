from pathlib import Path


# %%
def rename_files(directory):
    directory_path = Path(directory)
    for file_path in directory_path.rglob("*"):
        if file_path.is_file():
            new_filename = (
                file_path.name.replace("_Q_", "_Questions_")
                .replace("_A_", "_Answers_")
                .replace("_P_", "_Paragraph_")
                .replace("_F_", "_Feedback_")
                .replace("_T_", "_Title_")
            )
            new_filename = (
                new_filename.replace("_Q.", "_Questions.")
                .replace("_A.", "_Answers.")
                .replace("_P.", "_Paragraph.")
                .replace("_F.", "_Feedback.")
                .replace("_T.", "_Title.")
            )
            new_file_path = file_path.with_name(new_filename)
            file_path.rename(new_file_path)


directory = "processed_reports/full"
rename_files(directory)
