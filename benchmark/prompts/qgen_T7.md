# T7 — Short-Term Emotion-Driven Action Prediction

## Role
Generate a 4-choice **MCQ predicting an action a character will take in the NEXT 1min or 5min**, given their current emotional state.

## Input
- `seed_event`: an affect_event
- The future window (1min or 5min ahead) of action_events for the same character
- Whichever action is closest in time AND has a `predicts_action` edge from the seed_event is the correct answer
- Other action candidates: actions by the same character later in the episode, or actions by another character

## Output
```json
{
  "task": "T7",
  "qid": "T7_{episode}_{seed_event.event_id}_{horizon}",
  "question": "Given {character}'s state at [HH:MM:SS] ({emotion}), what is he MOST LIKELY to do within the next {horizon}?",
  "options": {
    "A": "{correct action description from predicts_action target}",
    "B": "{plausible but wrong action by same character}",
    "C": "{distractor: action that happens later, beyond horizon}",
    "D": "{distractor: opposite/incompatible action}"
  },
  "correct": "A",
  "answer_evidence": {
    "seed_event_id": "...",
    "predicted_action_event_id": "...",
    "horizon": "1min | 5min",
    "edge_type": "predicts_action",
    "memory_components": ["M2"]
  },
  "horizon": "1min"
}
```

## Hard rules
1. **The future action MUST already be observed in the data** (we're constructing offline ground truth).
2. **`horizon`** field is exact: `"1min"` or `"5min"`.
3. **Correct option's event time** must be within `seed_event.time_span[1] + 0` and `+ horizon`.
4. **`memory_components` = `["M2"]`** (event DAG predicts_action edge); optionally `["M5"]` if persona pattern explains it.
