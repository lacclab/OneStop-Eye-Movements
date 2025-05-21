
# Data Files and Variables

## Data Files

We release the following data files:

- **[questionnaire.json](https://osf.io/a765t):** Anonymized participant demographics and language history questionnaire.

- **[session_summary.csv](https://osf.io/yvu5w)**: Summary statistics and information on each participant's experiment session, such as reading comprehension accuracy and experiment duration.

- **`[preview]_b[batch]_l[list]_[subj_id].edf`**: The raw eye-tracking data for the entire eye-tracking session of each participant, in EDF format. (To be added).

- **`[preview]_b[batch]_l[list]_[subj_id].asc`**: Gaze location and additional features at 1ms intervals for the entire eye-tracking session of each participant, in ASCII format. (To be added).

- **Fixation Reports**: Eye movement features, experiment and trial information, and linguistic word properties aggregated at the level of individual fixations, in tab-separated CSV format. A separate report is available for each of the Interest Periods.

- **Interest Area Reports**: Eye movement features, experiment and trial information, and linguistic word properties aggregated at the word level, in tab-separated CSV format. A separate report is available for each of the Interest Periods.

### Fixation and Interest Area Report Files

| Interest Period       | Page  | Content                  | Fixation Report          | Interest Area Report     |
|-----------------------|-------|--------------------------|--------------------------|--------------------------|
| Title                 | Title | Article title            | `fixations_Title.csv`        | `ia_Title.csv`               |
| Question Preview      | 1     | Question                 | `fixations_Question_preview.csv`| `ia_Question_preview.csv`       |
| Paragraph             | 2     | Paragraph                | `fixations_Paragraph.csv`        | `ia_Paragraph.csv`               |
| Question              | 3     | Question                 | `fixations_Question.csv`        | `ia_Question.csv`               |
| Answers               | 4     | Question and answers     | `fixations_Answers.csv`        | `ia_Answers.csv`               |
| QA                    | 3+4   | Question and answers     | `fixations_QA.csv`       | `ia_QA.csv`              |
| Feedback              | 5     | Correct/Incorrect        | `fixations_Feedback.csv`        | `ia_Feedback.csv`               |

---

## Data Variables

### Participant Questionnaire Variables

Participant questionnaire variables. *Reading habits questions are based on Section 1 of the reading habits self-report of Acheson et al (2008).

| Variable                  | Description                                     | Values                                                   |
|---------------------------|-------------------------------------------------|---------------------------------------------------------|
| Participant ID            | Participant's ID                     | String (360 unique values)                                                  |
| Age                       | Participant's age                               | Years                                                    |
| Gender                    | Participant's gender                            | Male / Female / Other                                    |
| Home Country              | Participant's home country                      | List of countries                                        |
| Education Level           | Highest/current level of education              | Secondary, Undergraduate, Postgraduate                  |
| Native English Speaker    | Native English speaker                          | Yes / No                                                 |
| English AoA               | English Age of Acquisition                      | Since birth, or numeric age                             |
| Reading Habits            | Weekly time spent reading in various categories | 0 to 7+ hours                                           |
| Dyslexia                  | Presence of dyslexia                            | No / Dyslexia                                           |
| Language Impairments      | Language impairments                            | No / Impairment (free text)                             |
| Eye Conditions            | Eye conditions                                  | No / Specific condition (Amblyopia, etc.)              |

### Session Summary Variables

Session summary file variables.

| **Variable**                               | **Description**                                                                                         | **Values**                                                                                             |
|--------------------------------------------|---------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| participant_id                             | Participant's ID                                                                             | String (360 unique values)                                                                                                 |
| article_batch                              | A 10-article batch assigned to the participant                                                          | 1 (articles 1-10) / 2 (articles 11-20) / 3 (articles 21-30)                                            |
| list_number                                | Experimental list                                                                                       | 1 - 60                                                                                                 |
| question_preview                           | Was the question presented before the paragraph (i.e., the reading regime)                              | True / False                                                                                           |
| data_collection_site                       | Location of data collection                                                                             | MIT / Technion                                                                                         |
| comprehension_score-regular_trials         | Participant's overall reading comprehension score during first reading (10 articles, 54 regular trials) | 0-100%                                                                                                 |
| comprehension_score-repeated_reading       | Participant's overall reading comprehension score during repeated reading (2 articles, 8-14 repeated trials) | 0-100%                                                                                                 |
| recalibration_count                        | Number of times the session was interrupted to recalibrate the eye tracker                              | 0 or more                                                                                              |
| total_recalibrations                       | Number of times the eye tracker was recalibrated during the session (in addition to the 3 obligatory calibrations) | 0 or more                                                                                              |
| mean_validation_error                      | Mean validation error across all calibrations immediately preceding text reading                        | visual degrees                                                                                         |
| total_session_duration                     | Total duration of the experimental session (including breaks and calibrations)                          | minutes                                                                                                |
| session_duration                           | Duration of the experimental session excluding breaks and calibrations                                  | minutes                                                                                                |
| dominant_eye                               | Participant's dominant eye                                                                              | L / R                                                                                                  |
| tracked_eye                                | Eye that was tracked (typically the dominant eye)                                                      | L / R / LR **                                                                                          |
| lextale_score                             | Participant's score on the LexTALE vocabulary test *                                                     | 0 - 100                                                                                                |

**Notes:**

- *LexTale scores are available for 100 participants.
- **L: left eye, R: right eye, LR: data was collected from both eyes (switched between eyes during the experiment).

### Experiment, Trial, and Linguistic Annotation Variables

Experiment and trial variables, and linguistic annotations in the Fixation and Interest Area reports. UD annotations are extracted using spaCy. See the SR Data Viewer user manual for documentation of eye movement variables in these reports. Note, missing values are denoted by ".".

| **Category**                     | **Feature**                  | **Description**                                                                                                    | **Values**                                                                                           |
|-----------------------------------|------------------------------|--------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Experiment Variables**          | participant_id               | Participant's ID                                                                                        | String (360 unique values)                                                                                              |
|                                   | list_number                  | Experimental list                                                                                                  | 1 - 60                                                                                              |
|                                   | question_preview             | Was the question presented before the paragraph (i.e., the reading regime)                                         | True / False                                                                                        |
|                                   | article_batch                              | A 10-article batch assigned to the participant                                                          | 1 (articles 1-10) / 2 (articles 11-20) / 3 (articles 21-30)                                            |
| **Trial Variables**               | trial_index                  | The trial index                                                                                                    | 1 - 70                                                                                      |
|                                   | practice_trial               | Whether the trial was a practice trial                                                                             | True / False                                                                                        |
|                                   | article_id                   | The unique identifier for an article in a batch. Article 0 is the practice article                                                                    | 0-10                                                                                                |
|                                   | paragraph_id                 | The unique identifier for a paragraph in an article                                                                | 1-7                                                                                                 |
|                                   | difficulty_level             | Paragraph difficulty level                                                                                         | Adv / Ele                                                                                           |
|                                   | repeated_reading_trial       | Whether the trial was a repeated reading trial                                                                     | True / False                                                                                        |
|                                   | article_index                | The index of the article in the session. Article 0 is the practice article                                         | 0-12                                                                                                |
|                                   | article_title                | The article title, presented before the first paragraph of each article                                            | String                                                                                              |
|                                   | paragraph                    | The paragraph presented in the trial                                                                               | String                                                                                              |
|                                   | question                     | The question presented in the trial                                                                                | String                                                                                              |
|                                   | onestopqa_question_id                    | The unique identifier for the question presented in the trial                                                                                | 0-2                                                            |                                                                                             
|                                   | same_critical_span           | Whether there was another question with the same critical span                                                     | 0 if no other question. 1 or 2 otherwise (arbitrarily per question set).                            |
|                                   | selected_answer              | The answer selected by the participant                                                                             | A/B/C/D                                                                                             |
|                                   | selected_answer_position     | The position on the page of the answer selected by the participant                                                 | 0/1/2/3 corresponding to answer positions: top, left, right, bottom                                 |
|                                   | correct_answer_position      | The position on the page of the correct answer for the trial                                                       | 0/1/2/3 corresponding to answer positions: top, left, right, bottom                                 |
|                                   | answers_order                | Mapping between position on page and A/B/C/D                                                                       | list of ABCD corresponding to answer positions: top, left, right, bottom                            |
|                                   | answer_1                     | The answer presented in the trial in the top position                                                              | String                                                                                              |
|                                   | answer_2                     | The answer presented in the trial in the left position                                                             | String                                                                                              |
|                                   | answer_3                     | The answer presented in the trial in the right position                                                            | String                                                                                              |
|                                   | answer_4                     | The answer presented in the trial in the bottom position                                                           | String                                                                                              |
| **Linguistic Annotations - Big Three** | word_length                | Number of characters in the word                                                                                   | Integer                                                                                             |
|                                   | word_length_no_punctuation   | Number of characters in the word excluding punctuation                                                             | Integer                                                                                             |
|                                   | subtlex_frequency            | Log word frequency from the SUBTLEX-US database                                                                    | Bits                                                                                                |
|                                   | wordfreq_frequency           | Log word frequency from the Wordfreq database                                                                      | Bits                                                                                                |
|                                   | gpt2_surprisal               | Word surprisal extracted from the GPT-2 language model                                                             | Bits                                                                                                |
| **Linguistic Annotations - Universal Dependencies (UD)** | universal_pos                | Universal part-of-speech tag                                                                                       | 17 [possible tags](https://universaldependencies.org/u/pos)                                         |
|                                   | ptb_pos                      | Penn Treebank part-of-speech tag                                                                                   | See Label Scheme - TAGGER [here](https://spacy.io/models/en#en_core_web_sm)                         |
|                                   | head_word_index              | Index of the syntactic head word in the dependency tree                                                            | Integer, 0 for root - number of words in the sentence                                               |
|                                   | dependency_relation          | Dependency relation label to the head word in the dependency tree                                                  | See Label Scheme - PARSER [here](https://spacy.io/models/en#en_core_web_sm)                         |
|                                   | left_dependents_count        | Number of syntactic dependents to the left                                                                         | Integer                                                                                             |
|                                   | right_dependents_count       | Number of syntactic dependents to the right                                                                        | Integer                                                                                             |
|                                   | distance_to_head             | Distance in words to the syntactic head                                                                            | Integer, starting at 1 for adjacent words                                                          |
|                                   | morphological_features       | List of morphological features of the word                                                                         | See list [here](https://universaldependencies.org/u/feat/index.html)                                |
|                                   | entity_type                  | The entity type of the word (if applicable)                                                                        | See Label Scheme NER [here](https://spacy.io/models/en#en_core_web_sm). *None* if not an entity.   |
| **STARC Auxiliary Spans**         | auxiliary_span_type          | Whether a word is part of the critical span or the distractor span                                                 | critical / distractor / outside                                                                     |
|                                   | critical_span_indices        | Start and end word indices of the critical span                                                                    | list of tuples of integers                                                                          |
|                                   | distractor_span_indices      | Start and end word indices of the distractor span                                                                  | list of tuples of integers                                                                          |

Note that we remove the following SR variables from the reports as they are the same value:

IA Report Variables:

| **Variable**            | **Values**         |
|-------------------------|--------------------|
| GROUPING_VARIABLES      | RECORDING_SESSION  |
| IA_DYNAMIC              | False              |
| IA_END_TIME             | .                  |
| IA_GROUP                | .                  |
| IA_INSTANCES_COUNT      | .                  |
| IA_POINTS               | .                  |
| IA_START_TIME           | .                  |
| IA_TYPE                 | RECTANGLE          |
| IP_END_EVENT_MATCHED    | True               |
| IP_INDEX                | 1                  |
| IP_LABEL                | <SESSION_PAGE>     |
| IP_START_EVENT_MATCHED  | True               |
| REPORTING_METHOD        | Fixations          |
| TIME_SCALE              | Trial Relative     |
| DUMMY                   | NaN                |
| LETTER_HIGHT            | 38                 |
| LETTER_WIDTH            | 19                 |
| RECALIBRATE             | False              |
| TRIGGER_PADDING_X       | 10                 |
| Trial_Recycled_         | False              |
|X,Y answer coordinates   |{[(930, 414), (180, 756), (1680, 756), (930, 1098)]}|
