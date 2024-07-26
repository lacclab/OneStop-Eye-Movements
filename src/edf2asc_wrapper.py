from pathlib import Path
import subprocess

def find_edf_files(directory):
    edf_files = list(directory.rglob('*.edf'))
    return edf_files


def run_edf2asc(edf_file, output_dir):
    command = f'edf2asc "{edf_file}" -t -c -v -y -utf8 -p "{output_dir}"'
    with open('edf2asc.log', 'a') as f:
        subprocess.run(command, shell=True, check=False, stdout=f, stderr=subprocess.STDOUT)

if __name__ == '__main__':
    
    root_dir = Path("/Users/shubi/Data/OneStop Full Experiment Folders Backup 1932024 after fixes")
    output_dir = root_dir/'asc'
    output_dir.mkdir(parents=True, exist_ok=True)

    edf_files = find_edf_files(root_dir)

    if not edf_files:
        print('No .edf files found in the specified directory.')
    else:
        print(f'Found {len(edf_files)} .edf files.')
        for edf_file in edf_files:
            print(f'Processing {edf_file}...')
            run_edf2asc(edf_file, output_dir)
            print('Done.')