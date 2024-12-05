# Known Issues

This page lists known issues with the dataset. If you encounter any issues not listed here, please report them to the [issue tracker](https://github.com/lacclab/OneStop-Eye-Movements/issues).

Typos and cut-off text:

| Batch | Article Title (ID)                                             | Paragraph | Question | Issue                                                                                     |
|-------|----------------------------------------------------------------|-----------|----------|-------------------------------------------------------------------------------------------|
| 2     | Article 8                                                      | 5         | 2        | Typo: "culture"". instead of "culture".                                                   |
| 3     | John Lewis Christmas Advertisement 2015: Raising Awareness for Age UK (5) | 3         | 1        | Typo: "When was the hashtag” → "When was the hashtag #OnTheMoon released this year?”      |
| 3     | John Lewis Christmas Advertisement 2015: Raising Awareness for Age UK (5) | 3         | 2        | Typo: "Who used the” → "Who used the #OnTheMoon hashtag for their campaign?”              |
| -     | Google car article                                             | -         | -        | Typo: “food pedal” —> “foot pedal”                                                        |
| -     | Agios article                                                  | -         | -        | Typo: “fairies” —> “ferries”                                                              |
| -     | -                                                              | -         | -        | Possible typo: dangerous *president / precedent                                           |
| -     | Swarthy Blue-Eyed Caveman Revealed (6)                         | 2         | -        | Cut-off text: "Since the time of the hunter-gatherers, immune systems have been dependent on a health diet" appears in the ‘d’ position (bottom), the last word “diet” is cut off. |

Other issues:

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
