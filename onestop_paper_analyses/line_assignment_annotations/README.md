# Accuracy of fixation assignments to lines

This folder contains data for an analysis that quantifies the prevalence of erroneous line assignments for fixations due to vertical drift in OneStop Eye Movements.

## Annotation scheme

We divide fixations into three categories:

1. **Line-reading fixations assigned to the correct line**
2. **Line-reading fixations assigned to an incorrect line** (errors due to vertical drift)

   Line-reading fixations are part of fixation sequences (typically of at least three fixations) that exhibit an approximately horizontal progressive and/or regressive reading pattern.
3. **Other**

   Includes the following cases:
   - Return sweep completion sequences. These sequences start with the landing fixation of a return sweep saccade from the end of a line to the beginning of the next line, or the landing fixation of an abrupt, typically long, saccade from any location on a line to another line. They end with the last fixation before proceeding to the right on the new line to a sequence of at least two line reading fixations or with the last fixation of the trial. 
   - Fixations outside the trial's interest areas.
   - Fixations shorter than 50ms or longer than 500ms.

## Folder content

**`fixation_annotations.csv`** - Manual annotation of fixation categories for 20 trials with 10 lines each (1,296 fixations in total) by two annotators. See further details on trial selection criteria in the paper.

- `other_annotator[annotator_id]` fixations marked as other by an annotator.
- `incorrect_line_annotator[annotator_id]` fixations marked as line-reading assigned to an incorrect line by an annotator.
- `file_name` name of the file with the trial visualization in the `scanpath_visualizations/` folder.

**`scanpath_visualizations/`** - Scanpath visualizations of the 20 annotated trials.

- Each fixation is represented with a circle.
- In the center of each circle is the fixation number (the index in the trial's fixation sequence, starting with 0).
- Fixations are color-coded by their assigned line.
- Horizontal lines indicate line boundaries.
- Fixations shorter than 50ms or longer than 500ms are colored in blue.
