# Adjudication guide

Two people labelled the same documents independently and disagreed on some of them. You decide the correct labels for those documents.
The rules are in `docs/labeling-policy.md` and the label format in `docs/testset-labeler-guide.md`. Follow them exactly.
Do not read anything else in the repository except the files named below.

## Input and output

- a disagreements file (the path is given in your instructions): one row per disputed document with `sensitivity` (the two answers), `only_first` and `only_second` (findings only one of them listed).
- `eval/work/blind/<name>.json`: the texts.

Write the decisions file named in your instructions:

```json
{"tests": [{"id": "...", "expected": {"sensitivity": "none|low|high", "findings": [ ... ]}, "note": "<one short reason>"}]}
```

For each disputed document, read the whole text yourself and write the complete correct label set (all findings, not only the disputed ones). Neither side is presumed right; either or both may have missed something, so look for values neither of them listed. Apply the policy as written, in particular: names of public figures, historical figures and deceased persons are listed as names; a roster under a heading that reveals a sensitive fact is high; every category of sensitive information is judged by the same standard. Every `value` must be an exact substring of the text.

## Working method

Decide every document yourself by reading it. Do not decide with code, regular expressions or by merging the two answers mechanically. Use Python only to write your decisions into the JSON file and to check that it loads, covers every disputed id, and that every value is a substring of its text.
