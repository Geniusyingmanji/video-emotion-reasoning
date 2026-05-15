# T6 — Multi-Hop Emotion Causal Chain

## Role
Generate a 4-choice **MCQ asking which event CHAIN leads to a current emotional state**. Requires ≥3 hops.

## Input
- A `seed_event` (affect_event) deep in the episode/season
- A causal chain of ≥3 ancestor events traced via M2 / M3 edges
- 2-3 wrong chains as distractor candidates

## Output
```json
{
  "task": "T6",
  "qid": "T6_{episode}_{seed_event.event_id}",
  "question": "Which sequence of events most directly led to {character}'s state at [HH:MM:SS]?",
  "options": {
    "A": "ev_a → ev_b → ev_c → seed_event (correct, ≥3 hops along causal/trigger edges)",
    "B": "shorter chain that LOOKS plausible but skips a key event",
    "C": "chain with one edge replaced by a temporal-only edge (correlation not causation)",
    "D": "completely wrong chain involving a different subplot"
  },
  "correct": "A",
  "answer_evidence": {
    "chain": ["ev_a", "ev_b", "ev_c", "seed_event"],
    "edge_types_used": ["causal", "emotion_trigger", "..."],
    "memory_components": ["M2", "M3"]
  },
  "horizon": null
}
```

## Hard rules
1. **Chain length ≥3** for correct option.
2. **Chain edges must all exist** in M2 or M3 high-confidence edges.
3. **At least one distractor** must be a "plausible but incomplete" chain to test if model just picks shorter.
4. **`memory_components` = `["M2"]` or `["M2", "M3"]`** depending on chain crossing episodes.
