# T1 — Emotion Recognition

## Role
Generate a 4-choice **MCQ that requires identifying the emotion of a character at a specific moment**.

## Input (provided by pipeline)
- `seed_event`: an affect_event row (from events.jsonl) with character, emotion, intensity, time_span, cues, summary
- `nearby_perception`: the perception trace within ±10s of the event (utterances, paralinguistic, characters present)

## Output (strict JSON)
```json
{
  "task": "T1",
  "qid": "T1_{episode}_{seed_event.event_id}",
  "question": "At [HH:MM:SS], what is {character}'s primary emotion?",
  "options": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  },
  "correct": "A",
  "answer_evidence": {
    "event_id": "...",
    "time_span": [..., ...],
    "modalities_required": ["visual", "audio", "text"],
    "memory_components": ["M1"]
  },
  "horizon": null,
  "candidate_distractors": ["sadness", "anger", "fear", "joy"]
}
```

## Hard rules
1. The **correct emotion** comes from `seed_event.emotion`.
2. **Distractors** must be emotions that are plausibly confusable in the context but contradicted by at least one cue in the perception trace. Don't use trivially-wrong opposites only.
3. **`memory_components` MUST be `["M1"]`** for T1 — this is the local-only task.
4. Format the time as `HH:MM:SS` from the event start.
5. **Output strict JSON only**, no prose.
