# Adjudication guide

Two people labelled the same documents independently and disagreed on some of them. You decide the correct labels for those documents.
The rules are in `docs/labeling-policy.md` (Japanese) and the label format in `docs/testset-labeler-guide.md`. Follow them exactly.
Do not read anything else in the repository except the files named below.

## Input and output

- `eval/work/disagreements/<name>.json`: one row per disputed document with `sensitivity` (the two answers), `only_first` and `only_second` (findings only one of them listed).
- `eval/work/blind/<name>.json`: the texts.

Write `eval/work/decisions/<name>.json`:

```json
{"tests": [{"id": "...", "expected": {"sensitivity": "none|low|high", "findings": [ ... ]}, "note": "<one short reason>"}]}
```

For each disputed document, read the text yourself and write the complete correct label set (all findings, not only the disputed ones). Neither side is presumed right; either or both may have missed something. Every `value` must be an exact substring of the text.

## Working method

Decide every document yourself by reading it. Do not decide with code, regular expressions or by merging the two answers mechanically. Use Python only to write your decisions into the JSON file and to check that it loads, covers every disputed id, and that every value is a substring of its text.
