# OneStop: A 360-Participant Eye-Tracking Dataset for Reading Comprehension, Readability and Re-Reading in Ordinary and Information-Seeking Reading

We present OneStop, a corpus of eye movements in reading, in which native (L1) English speakers read texts in English from the Guardian and answer reading comprehension questions about them.

OneStop is the largest English eye-tracking dataset with L1 speakers to date, comprising eye movement recordings from 360 participants over 2.6 million word tokens.

The experiment is conducted using extensively piloted reading comprehension materials with 486 multiple choice reading comprehension questions and auxiliary text annotations geared towards behavioral analyses of reading comprehension.

OneStopGaze further includes controlled experimental manipulations of the difficulty level of the text, the presented questions, first reading versus re-reading, and ordinary reading versus information seeking.
<!--- The broad coverage and controlled experimental design of OneStopGaze aim to enable new research avenues in the cognitive study of reading and human language processing, and provide new possibilities for the integration of psycholinguistics with Natural Language Processing (NLP) and Artificial Intelligence (AI). --->

## Example

![Trial GIF](l6_187_Trial_20.gif)

## Obtaining the Data

There are several ways to obtain the data:

1. Download the data from the [OSF repository](https://osf.io/8z2xv/).
2. Use the `download_data_files.py` script to download and extract the data files automatically.
3. Use the `pymovements` package to download the data.
4. TODO Add other ways to obtain the data?

### Direct Download from OSF

The data files are stored in an [OSF repository]() (TODO add link), and can be downloaded manually from the repository.

### Python Script

If the repository has been cloned, they can be downloaded and extracted automatically using the following script:

```python
# or python3
python download_data_files.py

# OR to extract the files directly
python download_data_files.py --extract
Alternatively, they can be downloaded manually from the OSF repository and extracted into the respective folders.
```

TODO Update the scripts

### pymovements integration

OneStop is integrated into the [pymovements](https://pymovements.readthedocs.io/en/stable/index.html) package. The package allows to easily download the raw data and further process it. The following code snippet shows how to download the data:

```python
# pip install pymovements
import pymovements as pm

dataset = pm.Dataset('PoTeC', path='data/PoTeC')

dataset.download()
```

## Statistics

| **Subjs** | **Age**        | **Words**      | **Words Recorded** | **Qs** | **Subjs per Q** | **Qs per Subj** |
|----------|----------------|----------------|--------------------|--------|-----------------|-----------------|
| 360       | 23.0 ($\pm$6.4)| 21,384 (Adv); 16,817 (Ele)   | 2,631,563          | 486    | 20              | 54              |

TODO Copy values from paper.

## Directory Structure

**`data_[version]/`**

SR DataViewer Interest Area and Fixation Reports, and syntactic annotations.

- `sent_ia.tsv` Interest Area report.  
- `sent_fix.tsv` Fixations report.
- `annotations/` Syntactic annotations.

**`participant_metadata/`**

- `metadata.tsv` metadata on participants.
- `languages.tsv` information on languages spoken besides English.
- `test_scores/`
  - `test_conversion.tsv` unofficial conversion table between standardized proficiency tests (used to convert TOEIC to TOEFL scores).
  - `michigan-cefr.tsv` conversion table between form B and the newer forms D/E/F, as well as to CEFR levels.
  - `michigan/` item level responses for the Michigan Placement Test (MPT).
  - `comprehension/` item level responses for the reading comprehension during the eyetracking experiment.  

**`splits/`**

Trial and participant splits.

- `trials/`
  - `all_trials.txt` trial numbers for all the sentences (1-157).
  - `shared_trials.txt` trial numbers of the Shared Text regime.
  - `individual_trials.txt` trial number of the Individual Text regime.
- `participants/[version]/`
  - `random_order.csv` random participant order.
  - `train.csv` train participants.
  - `test.csv` test participants.

<a name="docs">

**`dataset_analyses.Rmd`**

Analyses for the paper "CELER: A 365-Participant Corpus of Eye Movements in L1 and L2 English Reading".
Note that this script requires:

- CELER (in the folder `data_v2.0/`) and,
- GECO Augmented (in the folder `geco/`). Download [GECO augmented](https://drive.google.com/file/d/1T4qgbwPkdzYmTvIqMUGJlvY-v22Ifinx/view?usp=sharing) with frequency and surprisal values and place `geco/` at the top level of this directory.

## Documentation

</a>

- [Eyetracking Variables](documentation/data_variables.md) Description of the variables in the fixations and interest area reports.
- [Metadata Variables](documentation/metadata_variables.md) Description of the variables in the participants metadata and languages files.
- [Language Models](documentation/language_models.md) Details on language models for surprisal values.
- [Syntactic Annotations](documentation/syntactic_annotations.md) Details on syntactic annotations (POS, phra`se structure trees, dependency trees).
- [GECO Augmented](documentation/geco_augmented.md) Details on new fields added to GECO.
- [Experiment Builder Programs](documentation/EB_programs.md) Information on the EB experiment.
- [Known Issues](documentation/known_issues.md) Known issues with the dataset.


## Citation

Paper: TODO

```
TODO

```

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

![Creative Commons License](https://i.creativecommons.org/l/by/4.0/88x31.png)

TODO license okay?
