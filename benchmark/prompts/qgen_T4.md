# T4 — Emotion Shift Localization

## Role
Generate a 4-choice **MCQ asking when an emotion shift happens** (which event/timestamp marks the transition).

## Input
- A pair of adjacent affect_events (same character, different emotion).
- The intervening events between them.

## Output
```json
{
  "task": "T4",
  "qid": "T4_{episode}_{evA.event_id}_{evB.event_id}",
  "question": "When does {character}'s emotion shift from {emoA} to {emoB}?",
  "options": {
    "A": "Around [HH:MM:SS] (correct)",
    "B": "Around [HH:MM:SS] (distractor: ±60s)",
    "C": "Around [HH:MM:SS] (distractor: from another character's similar shift)",
    "D": "Around [HH:MM:SS] (distractor: well before/after)"
  },
  "correct": "A",
  "answer_evidence": {
    "event_id": "...",
    "trigger_event_id": "...",
    "time_span": [..., ...],
    "memory_components": ["M1", "M2"]
  },
  "horizon": null
}
```

## Hard rules
1. **Time precision**: options should be within ±60s of the actual shift.
2. The correct answer is `evB.time_span[0]` (where the new emotion begins).
3. **`memory_components` = `["M1", "M2"]`** — requires shot-level perception AND the event DAG to identify the shift.
4. **Output strict JSON only**.
