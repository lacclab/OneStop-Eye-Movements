# Known Issues

This page lists known issues with the dataset. If you encounter any issues not listed here, please report them to the [issue tracker](https://github.com/lacclab/OneStop-Eye-Movements/issues).

Typos and cut-off text:

| Batch | Article Title                                      | Article ID | Paragraph | Question | Issue                                                                                     |
|-------|----------------------------------------------------|------------|-----------|----------|-------------------------------------------------------------------------------------------|
| 2     | Rwandan Women Whip up Popular Ice Cream Business                                            | 8          | 5         | 2        | Typo: "culture"". instead of "culture".  TODO WHERE?                                                   |
| 3     | John Lewis Christmas Advertisement 2015: Raising Awareness for Age UK | 5          | 3         | 1        | Typo in the question: "When was the hashtag” instead of "When was the hashtag #OnTheMoon released this year?” TODO WHEN WAS THIS FIXED?      |
| 3     | John Lewis Christmas Advertisement 2015: Raising Awareness for Age UK | 5          | 3         | 2        | Typo in the question: "Who used the” instead of "Who used the #OnTheMoon hashtag for their campaign?” TODO WHEN WAS THIS FIXED?              |
| 1     | Google Introduces its Driverless Car               | 1          | 1        | 2        | Typo in Answer B: “food pedal” instead of “foot pedal”.                                                        |
| 1     | The Greek Island Where Time Is Running Out                                      | 10          | 6         | 2        | Typo in Answer D: “fairies” instead of “ferries”                                                              |
| -     | -                                                  | -          | -         | -        | TODO CHECK OUT Possible typo: dangerous *president / precedent                                           |
| 1     | Swarthy Blue-Eyed Caveman Revealed                 | 6          | 2         | 0        | Cut-off text: When Answer D "Since the time of the hunter-gatherers, immune systems have been dependent on a health diet" appears in the bottom position, the last word, “diet”, is cut off. |

Other issues TODO Go Over:

- Near-sightedness / Myopia was not in the eye conditions list and not everyone who had it necessarily marked it.
- There are 2 missing trials:
  - l13_213 missed trial 39  (`paragraph_id=4` of `article_id=10`)
  - l34_277 missed  trial 38 (`paragraph_id=5` of `article_id=2`)
- Pixel issue
- The following raw critical and distractor span annotations are incorrect and can be fixed using the processed files.

    | Article Title                                               | Paragraph ID | Level | Q Index | Column       | Corrected Value  |
    |-------------------------------------------------------------|--------------|-------|---------|--------------|------------------|
    | Japan Calls Time on Long Hours Work Culture                 | 3            | Adv   | 2       | dspan_inds   | [(79, 102)]      |
    | Japan Calls Time on Long Hours Work Culture                 | 3            | Ele   | 2       | dspan_inds   | [(64, 80)]       |
    | Love Hormone Helps Autistic Children Bond with Others       | 6            | Adv   | 1       | aspan_inds   | [(49, 67)]       |
