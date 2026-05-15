# T5 — Single-hop Emotion Cause Attribution

## Role
Generate a 4-choice **MCQ asking what most directly caused a character's emotion at a specific moment**.

## Input
- A `seed_event` (affect_event)
- Its `trigger_ref` (the cause event in M2)
- 2-3 other temporally-near events that are NOT the cause (for distractors)

## Output
```json
{
  "task": "T5",
  "qid": "T5_{episode}_{seed_event.event_id}",
  "question": "What most directly caused {character}'s {emotion} at [HH:MM:SS]?",
  "options": {
    "A": "{cause_event summary} (correct, from trigger_ref)",
    "B": "{distractor: temporally close but causally unrelated event}",
    "C": "{distractor: same character but earlier emotion}",
    "D": "{distractor: dialogue that LOOKS like a cause but isn't}"
  },
  "correct": "A",
  "answer_evidence": {
    "event_id": "...",
    "trigger_event_id": "...",
    "edge_type": "emotion_trigger",
    "memory_components": ["M2"]
  },
  "horizon": null
}
```

## Hard rules
1. **Correct = the cause event whose edge type is `causal` or `emotion_trigger`** in M2.
2. **Distractors must be plausible**: same time window, same character, OR similar-sounding-but-wrong dialogue.
3. **`memory_components` = `["M2"]`** at minimum (event DAG required); optionally + `["M3", "M5"]` if cross-ep / persona context strengthens the answer.
