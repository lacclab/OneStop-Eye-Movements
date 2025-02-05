# Known Issues

This page lists known issues with the dataset. If you encounter any issues not listed here, please report them to the [issue tracker](https://github.com/lacclab/OneStop-Eye-Movements/issues).

Typos and cut-off text:

| Batch | Article Title                                      | Article ID | Paragraph | Question | Issue                                                                                     |
|-------|----------------------------------------------------|------------|-----------|----------|-------------------------------------------------------------------------------------------|
| 2     | Rwandan Women Whip up Popular Ice Cream Business                                            | 8          | 5 (Adv)        | -        | Typo: _culture""._ instead of _culture"._  for ['l11_205', 'l13_206', 'l13_213', 'l15_210', 'l15_214', 'l17_219', 'l19_225', 'l19_230', 'l1_155', 'l1_161', 'l21_227', 'l21_232', 'l23_234', 'l23_235', 'l25_238', 'l25_243', 'l27_245', 'l27_270', 'l29_247', 'l29_248', 'l31_251', 'l31_257', 'l33_258', 'l33_276', 'l35_264', 'l35_278', 'l37_273', 'l39_289', 'l39_299', 'l3_163', 'l3_167', 'l41_295', 'l41_303', 'l43_302', 'l43_310', 'l45_306', 'l47_313', 'l47_319', 'l5_175', 'l5_179', 'l7_183', 'l7_199', 'l9_195', 'l9_201']                                                 |
| 3     | John Lewis Christmas Advertisement 2015: Raising Awareness for Age UK | 5          | 3         | 1        | Cutoff question: "When was the hashtag” instead of "When was the hashtag #OnTheMoon released this year?” Only for participants `l3_325, l2_323, l3_327, l2_324`.     |
| 3     | John Lewis Christmas Advertisement 2015: Raising Awareness for Age UK | 5          | 3         | 2        | Cutoff question: "Who used the” instead of "Who used the #OnTheMoon hashtag for their campaign?” Only for participants `l4_326` and `l5_328`.         |
| 1     | Google Introduces its Driverless Car               | 1          | 1        | 2        | Typo in Answer B: “food pedal” instead of “foot pedal”.                                                        |
| 1     | The Greek Island Where Time Is Running Out                                      | 10          | 6         | 2        | Typo in Answer D: “fairies” instead of “ferries”                                                              |
| 1     | Swarthy Blue-Eyed Caveman Revealed                 | 6          | 2         | 0        | Cut-off text: When Answer D "Since the time of the hunter-gatherers, immune systems have been dependent on a health diet" appears in the bottom position, the last word, “diet”, is cut off. |

Other issues:

- Near-sightedness / Myopia was not in the eye conditions list and not everyone who had it necessarily marked it.
- There are 2 missing trials in the dataset:
  - l13_213 missed trial 39  (`paragraph_id=4` of `article_id=10`)
  - l34_277 missed  trial 38 (`paragraph_id=5` of `article_id=2`)
- Due to a software bug, some trials had a screen width of 1 pixel higher than others. This resulted in slightly different text position in some paragraphs in some trials. In some cases this resulted in 11 lines of text instead of 10. Using `fix_ias.ipynb` we manually created the missing row of ias in the `.ias` files for these trials. Note, in cases where a hyphenated word was split across two lines this results in two different versions of text. TODO Add code snippet to show how to find these.
- The following raw critical and distractor span annotations are incorrect in the raw files and are fixed in the processed files.

    | Article Title                                               | Paragraph ID | Level | Q Index | Column       | Corrected Value  |
    |-------------------------------------------------------------|--------------|-------|---------|--------------|------------------|
    | Japan Calls Time on Long Hours Work Culture                 | 3            | Adv   | 2       | dspan_inds   | [(79, 102)]      |
    | Japan Calls Time on Long Hours Work Culture                 | 3            | Ele   | 2       | dspan_inds   | [(64, 80)]       |
    | Love Hormone Helps Autistic Children Bond with Others       | 6            | Adv   | 1       | aspan_inds   | [(49, 67)]       |

Fixed issues:

- Fixed: SR-generated `.ias` files incorrectly labeled split hyphenated words (e.g., "hunter-" and "gatherer" got full "hunter-gatherer" label). Now each part gets its correct segment via `fix_ias.ipynb`.
- For some reason, the `question` field in reports was not always correctly populated (last word or two were cutoff), even though the participants saw the full question. This was fixed by manually adding the missing words to the `question` field for the following cases:

  | Batch | Article ID | Paragraph ID | Question Index |
  |-------|------------|--------------|----------------|
  | 1     | 1          | 7            | 2              |
  | 1     | 2          | 6            | 1              |
  | 1     | 8          | 2            | 1              |
  | 1     | 9          | 4            | 1              |
  | 1     | 9          | 5            | 0              |
  | 3     | 2          | 2            | 1              |
  | 3     | 3          | 2            | 2              |
