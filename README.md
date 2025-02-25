Given a JSONL file with a particular structure, this provides a way to visualize language model outputs, eg: 

![](https://i.imgur.com/Bb1v2Dk.png)

We expect a JSONL file where each line is a JSON object with the following keys:

- `tokens`: list of tokens (strings)
- `metrics`: dictionary of metric_name → list of float values with which to annotate the tokens (with the same length as `tokens`)
- `next_tokens` (optional): list of dictionaries (same length as `tokens`), which each map from a possible next token to its associated probability (or logits)
- `annotations` (optional): list of span annotations, each with:
  - `start`: starting token index
  - `end`: ending token index (exclusive)
  - `label`: annotation text
