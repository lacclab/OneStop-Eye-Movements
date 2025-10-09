# Preproccessing and Reproducing Analyses

Preproccessing and reproducing analyses in the paper "OneStop: A 360-Participant English Eye-Tracking Dataset with Different Reading Regimes":

## Environment Setup

### Prerequisites

* [Mamba](https://github.com/conda-forge/miniforge#mambaforge) or Conda

### Setup

1. **Clone the Repository**

    Start by cloning the repository to your local machine:

    ```bash
    git clone https://github.com/lacclab/OneStop-Eye-Movements.git
    cd OneStop-Eye-Movements
    ```

2. **Create a Virtual Environment**

    Create a new virtual environment using Mamba (or Conda) and install the dependencies:

    ```bash
    mamba env create -f environment.yaml
    ```

3. **Activate the Virtual Environment**

    Activate the virtual environment:

    ```bash
    conda activate onestop
    ```

4. **Download spacy model**

    Download the spacy model:

    ```bash
    python -m spacy download en_core_web_sm
    ```

## Preprocessing Data

The preprocess steps can be found in [process_sr_report.py](https://github.com/lacclab/OneStop-Eye-Movements/blob/main/data_preprocessing/process_sr_report.py) and [compute_session_summary.py](https://github.com/lacclab/OneStop-Eye-Movements/blob/main/data_preprocessing/compute_session_summary.py).

## Running the Analyses

The analyses in the OneStop paper can be reproduced by running [paper_analyses.ipynb](https://github.com/lacclab/OneStop-Eye-Movements/blob/main/onestop_paper_analyses/paper_analyses.ipynb).
