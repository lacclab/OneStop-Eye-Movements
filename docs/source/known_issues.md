# Known Issues

This page lists known issues with the dataset. If you encounter any issues not listed here, please report them to the [issue tracker](https://github.com/lacclab/OneStop-Eye-Movements/issues).

- Difference in. batch 3 - diff .dat files before and after fix:

    Batch 3,Article ID 5 (John Lewis Christmas Advertisement 2015: Raising Awareness for Age UK), Paragraph 3

    1. Q (q_ind) 1: "When was the hashtag” → "When was the hashtag #OnTheMoon released this year?”
    2. Q (q_ind) 2: "Who used the” → "Who used the #OnTheMoon hashtag for their campaign?”

- Google car article: “food pedal” —> “foot pedal”
- Agios article: “fairies” —> “ferries”
- Another possible typo in the data (found this on a note, haven’t checked): dangerous *president / precedent

- Article  Swarthy Blue-Eyed Caveman Revealed (Id 6) paragraph 2 when answer “Since the time of the hunter-gatherers, immune systems have been dependent on a health diet” appears in the ‘d’ position (bottom), the last word “diet” is cut off.

- The following raw annotations are incorrect and can be fixed using the processed files or the following code:

    ```python
    data.loc[
            (data["article_title"] == "Japan Calls Time on Long Hours Work Culture")
            & (data["paragraph_id"] == 3)
            & (data["level"] == "Adv")
            & (data["q_ind"] == 2),
            ["dspan_inds"],
        ] = "[(79, 102)]"

        data.loc[
            (data["article_title"] == "Japan Calls Time on Long Hours Work Culture")
            & (data["paragraph_id"] == 3)
            & (data["level"] == "Ele")
            & (data["q_ind"] == 2),
            ["dspan_inds"],
        ] = "[(64, 80)]"

        data.loc[
            (
                data["article_title"]
                == "Love Hormone Helps Autistic Children Bond with Others"
            )
            & (data["paragraph_id"] == 6)
            & (data["level"] == "Adv")
            & (data["q_ind"] == 1),
            "aspan_inds",
        ] = "[(49, 67)]"
    ```

- culture"". instead of culture".  in "level == 'Adv' and batch =='2' and article_id =='8' and paragraph_id=='5'”

- Near-sightedness / Myopia was not in the eye conditions list and not everyone who had it necessarily marked it.

2 missing trials:
- l13_213 missed trial 39  `paragraph_id=4` of `article_id=10`
- l34_277 missed  trial 38`paragraph_id=5` of `article_id=2`
