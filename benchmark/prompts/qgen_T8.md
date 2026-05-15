# T8 — Cross-Episode Action Prediction Conditioned on Emotion

## Role
Generate a 4-choice **MCQ asking what action a character will take in a FUTURE episode** as a consequence of an emotional event in the current episode.

## Input
- `seed_event`: an affect_event in episode N
- A future action_event in episode N+k (where k ≥ 1) that has a `predicts_action_cross_ep` or `long_causal` edge to seed_event in M3
- Distractor candidates: actions that happen later but are unrelated

## Output
```json
{
  "task": "T8",
  "qid": "T8_{ep_seed}_{seed_event.event_id}",
  "question": "Given {character}'s {emotion} after the events of [ep N event], what action is he most likely to take in a later episode?",
  "options": {
    "A": "{correct future action, from M3 cross-ep edge}",
    "B": "{plausible but unrelated future action}",
    "C": "{action that contradicts the emotional trajectory}",
    "D": "{action by a different character}"
  },
  "correct": "A",
  "answer_evidence": {
    "seed_event_id": "...",
    "predicted_action_event_id": "...",
    "src_episode": "epN",
    "dst_episode": "epM",
    "horizon": "30min | cross_episode",
    "edge_type": "predicts_action_cross_ep | long_causal",
    "memory_components": ["M3", "M4"]
  },
  "horizon": "cross_episode"
}
```

## Hard rules
1. **Source and target events MUST be in different episodes**.
2. **Correct answer must rely on M3 edge** — explicitly cite which cross-ep edge.
3. **`memory_components` MUST include both `M3` and `M4`** (cumulative trajectory matters).
4. **`horizon`** is `"30min"` or `"cross_episode"`.
