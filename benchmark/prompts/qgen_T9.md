# T9 — Personality (OCEAN) Consistency Prediction

## Role
Generate a 4-choice **MCQ asking what a character would do in a hypothetical/observed situation, given their OCEAN profile (M5)**. Tests whether reasoning is consistent with the persona.

## Input
- `character`: a character with a distilled M5 persona
- `situation`: an actual scene from the season (use an observed action_event as the ground-truth "what they did")
- `M5_persona`: their OCEAN profile + signature behaviors + backstory facts

## Output
```json
{
  "task": "T9",
  "qid": "T9_{character}_{action_event.event_id}",
  "question": "Based on what you know about {character}, when faced with [situation description], what did/would he do?",
  "options": {
    "A": "{actually-observed action, from M5-consistent action_event}",
    "B": "{personality-inconsistent action: violates a strong OCEAN trait}",
    "C": "{generic-sounding action: applicable to anyone, not personality-driven}",
    "D": "{action that fits a DIFFERENT character's persona}"
  },
  "correct": "A",
  "answer_evidence": {
    "character": "...",
    "action_event_id": "...",
    "ocean_traits_invoked": ["conscientiousness", "neuroticism"],
    "evidence_refs": ["..."],  // from M5
    "memory_components": ["M5"]
  },
  "horizon": null
}
```

## Hard rules
1. **`evidence_refs` MUST point to specific events in M5's evidence list** — no hallucination.
2. **Distractor B (inconsistent action) must violate a documented trait** with M5 score ≥ 0.6 or ≤ 0.4.
3. **Correct option must align with ≥2 OCEAN dimensions** from M5.
4. **`memory_components` = `["M5"]`** at minimum. Optionally + `["M6"]` if relational context informs the answer.
5. **Output strict JSON only**.
