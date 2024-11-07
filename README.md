# OneStop: A 360-Participant Eye-Tracking Dataset for Reading Comprehension, Readability and Re-Reading in Ordinary and Information-Seeking Reading

:::{admonition} OneStop
:class: note
We present OneStop, a corpus of eye movements in reading, in which native (L1) English speakers read texts in English from the Guardian and answer reading comprehension questions about them.
:::

## Example

![Trial GIF](_static/trial.gif)

## Overview

OneStop is the largest English eye-tracking dataset with L1 speakers to date, comprising eye movement recordings from 360 participants over 2.6 million word tokens.

The experiment is conducted using extensively piloted reading comprehension materials with 486 multiple choice reading comprehension questions and auxiliary text annotations geared towards behavioral analyses of reading comprehension.

OneStop further includes controlled experimental manipulations of the difficulty level of the text, the presented questions, first reading versus re-reading, and ordinary reading versus information seeking.
<!--- The broad coverage and controlled experimental design of OneStopGaze aim to enable new research avenues in the cognitive study of reading and human language processing, and provide new possibilities for the integration of psycholinguistics with Natural Language Processing (NLP) and Artificial Intelligence (AI). --->

## Obtaining the Data


<details>

<summary>Data will be available soon!</summary>

Data is not yet available. Download instructions will be updated here.

> There are several ways to obtain the data

> 1. Download the data directly from the [OSF repository](https://osf.io/2prdq/).
> 2. Use the `download_data_files.py` script to download and extract the data files automatically.
<!-- 3. Use the `pymovements` package to download the data. -->
<!-- 4. TODO Add other ways to obtain the data? -->

> ### Direct Download from OSF

> The data files are stored in an [OSF repository](https://osf.io/2prdq/), and can be downloaded manually from the repository.

> ### Python Script

> If the repository has been cloned, they can be downloaded and extracted automatically using the following script:

> ```python
> python download_data_files.py --extract
> ```

<!-- ### pymovements integration

OneStop is integrated into the [pymovements](https://pymovements.readthedocs.io/en/stable/index.html) package. The package allows to easily download the raw data and further process it. The following code snippet shows how to download the data:

```python
# pip install pymovements
import pymovements as pm

dataset = pm.Dataset('OneStop', path='data/OneStop')

dataset.download()
``` -->
</details>

## Repository Structure

The repository is structured as follows:

- `data/`: Contains the metadata files.
- `docs/`: Documentation source files.
- `src/`: Scripts to preprocess the data and to reproduce the analyses.

## Reproducing the Analyses

### Environment Setup

#### Prerequisites

- [Mamba](https://github.com/conda-forge/miniforge#mambaforge) or Conda

#### Setup

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

## Documentation

- [Variables](variables.md) Description of the variables in the dataset.
<!-- - [Eyetracking Variables](data_variables.md) Description of the variables in the fixations and interest area reports, including variables related to word properties and syntactic annotation.
- [Metadata Variables](metadata_variables.md) Description of the variables in the participants metadata. -->
- [Technical Setup](technical_setup.md) Information on the technical setup of the experiment and the experiment builder.
- [Known Issues](known_issues.md) Known issues with the dataset.

## Citation

Paper: TODO

```bash
@article{TODO,
  title={TODO},
  author={TODO},
  journal={TODO},
  year={TODO},
  volume={TODO},
  number={TODO},
  pages={TODO},
  doi={TODO}
}
```

## License

The code is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

![Creative Commons License](https://i.creativecommons.org/l/by/4.0/88x31.png)
