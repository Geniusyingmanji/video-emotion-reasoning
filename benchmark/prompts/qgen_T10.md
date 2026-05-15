# T10 — Emotion Contagion (Interpersonal Influence)

## Role
Generate a 4-choice **MCQ asking how Character B's emotion or action changes after Character A's emotional event**. Tests directional emotion contagion through the M6 relationship network.

## Input
- `source_event`: an affect_event for Character A
- `target_char`: a character with a relationship to A in M6
- The next ~5 min of perception including target_char's reactions (affect / action events)

## Output
```json
{
  "task": "T10",
  "qid": "T10_{source_event.event_id}_{target_char}",
  "question": "Right after {char_A}'s {emoA} at [HH:MM:SS], how does {char_B} respond emotionally and/or behaviorally?",
  "options": {
    "A": "{actually-observed emotional/behavioral response, from M4_B trajectory + post-source events}",
    "B": "{response that ignores the relationship type (e.g., comfort from an adversary)}",
    "C": "{response in opposite direction from observed (e.g., joy when actual is concern)}",
    "D": "{response from a different character, attributed to char_B by mistake}"
  },
  "correct": "A",
  "answer_evidence": {
    "source_char": "...",
    "source_event_id": "...",
    "target_char": "...",
    "response_event_ids": ["..."],
    "relation_type": "...",  // from M6
    "directionality": "directed | mutual",
    "memory_components": ["M4", "M6"]
  },
  "horizon": null
}
```

## Hard rules
1. **The relation between source and target must exist in M6**.
2. **Response must be observed in the data**, not hypothetical (we're constructing GT from data).
3. **Directionality is important**: ensure A → B, not B → A.
4. **`memory_components` = `["M4", "M6"]`** — both trajectories and relation network needed.
5. **Distractor B specifically tests that the model uses M6**: pick a response that would be plausible for a different relation type.
