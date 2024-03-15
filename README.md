# OneStop: A 360-Participant Eye-Tracking Dataset for Reading Comprehension, Readability and Re-Reading in Ordinary and Information-Seeking Reading

:::{note}
We present OneStop, a corpus of eye movements in reading, in which native (L1) English speakers read texts in English from the Guardian and answer reading comprehension questions about them.
:::
OneStop is the largest English eye-tracking dataset with L1 speakers to date, comprising eye movement recordings from 360 participants over 2.6 million word tokens.

The experiment is conducted using extensively piloted reading comprehension materials with 486 multiple choice reading comprehension questions and auxiliary text annotations geared towards behavioral analyses of reading comprehension.

OneStopGaze further includes controlled experimental manipulations of the difficulty level of the text, the presented questions, first reading versus re-reading, and ordinary reading versus information seeking.
<!--- The broad coverage and controlled experimental design of OneStopGaze aim to enable new research avenues in the cognitive study of reading and human language processing, and provide new possibilities for the integration of psycholinguistics with Natural Language Processing (NLP) and Artificial Intelligence (AI). --->

## Example

![Trial GIF](_static/trial.gif)

## Obtaining the Data

There are several ways to obtain the data:

1. Download the data from the [OSF repository](https://osf.io/8z2xv/).
2. Use the `download_data_files.py` script to download and extract the data files automatically.
3. Use the `pymovements` package to download the data.
4. TODO Add other ways to obtain the data?

### Direct Download from OSF

The data files are stored in an [OSF repository](TODO add link), and can be downloaded manually from the repository.

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


## Citation

Paper: TODO

```bash
TODO
```

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

![Creative Commons License](https://i.creativecommons.org/l/by/4.0/88x31.png)

TODO license okay?
