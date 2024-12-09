# Known Issues

This page lists known issues with the dataset. If you encounter any issues not listed here, please report them to the [issue tracker](https://github.com/lacclab/OneStop-Eye-Movements/issues).

Typos and cut-off text:

| Batch | Article Title                                      | Article ID | Paragraph | Question | Issue                                                                                     |
|-------|----------------------------------------------------|------------|-----------|----------|-------------------------------------------------------------------------------------------|
| 2     | Rwandan Women Whip up Popular Ice Cream Business                                            | 8          | 5         | 2        | Typo: "culture"". instead of "culture".  TODO WHERE?                                                   |
| 3     | John Lewis Christmas Advertisement 2015: Raising Awareness for Age UK | 5          | 3         | 1        | Cutoff question: "When was the hashtag” instead of "When was the hashtag #OnTheMoon released this year?” TODO Fixed for some? How is this marked? Add here. Look also at orange colored text. Check also  #OnTheMoon in text     |
| 3     | John Lewis Christmas Advertisement 2015: Raising Awareness for Age UK | 5          | 3         | 2        | Cutoff question: "Who used the” instead of "Who used the #OnTheMoon hashtag for their campaign?” TODO Fixed for some? How is this marked? Check also  #OnTheMoon in text              |
| 1     | Google Introduces its Driverless Car               | 1          | 1        | 2        | Typo in Answer B: “food pedal” instead of “foot pedal”.                                                        |
| 1     | The Greek Island Where Time Is Running Out                                      | 10          | 6         | 2        | Typo in Answer D: “fairies” instead of “ferries”                                                              |
| 1     | Swarthy Blue-Eyed Caveman Revealed                 | 6          | 2         | 0        | Cut-off text: When Answer D "Since the time of the hunter-gatherers, immune systems have been dependent on a health diet" appears in the bottom position, the last word, “diet”, is cut off. |

Other issues:

- Near-sightedness / Myopia was not in the eye conditions list and not everyone who had it necessarily marked it.
- There are 2 missing trials in the dataset:
  - l13_213 missed trial 39  (`paragraph_id=4` of `article_id=10`)
  - l34_277 missed  trial 38 (`paragraph_id=5` of `article_id=2`)
- Due to a software bug, some trials had a screen width of 1 pixel higher than others. This resulted in slightly different text position in some paragraphs in some trials. TODO How do we mark this?
- The following raw critical and distractor span annotations are incorrect in the raw files and are fixed in the processed files.

    | Article Title                                               | Paragraph ID | Level | Q Index | Column       | Corrected Value  |
    |-------------------------------------------------------------|--------------|-------|---------|--------------|------------------|
    | Japan Calls Time on Long Hours Work Culture                 | 3            | Adv   | 2       | dspan_inds   | [(79, 102)]      |
    | Japan Calls Time on Long Hours Work Culture                 | 3            | Ele   | 2       | dspan_inds   | [(64, 80)]       |
    | Love Hormone Helps Autistic Children Bond with Others       | 6            | Adv   | 1       | aspan_inds   | [(49, 67)]       |
