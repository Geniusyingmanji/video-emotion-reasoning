# T2 — Multimodal Evidence Grounding

## Role
Generate a 4-choice **MCQ asking which line of evidence best supports a given emotional claim**. Tests cross-modal fine-grained perception.

## Input
- `seed_event`: affect_event with emotion
- Its `cues` list (visual, audio, text cues at specific timestamps)

## Output (strict JSON)
```json
{
  "task": "T2",
  "qid": "T2_{episode}_{seed_event.event_id}",
  "question": "Which observation BEST shows that {character} is feeling {emotion} at [HH:MM:SS]?",
  "options": {
    "A": "He averts his eyes and his shoulders drop (visual, 14:23-14:25)",
    "B": "He raises his voice into a shout (audio, 14:25-14:28)",
    "C": "He says 'I don't care anymore' (text, 14:28)",
    "D": "Light flickers in the background (background motion, 14:23-14:30)"
  },
  "correct": "A",
  "answer_evidence": {
    "event_id": "...",
    "modalities_required": ["visual"],  // the modality of the correct cue
    "memory_components": ["M1"]
  },
  "horizon": null
}
```

## Hard rules
1. **One option per modality** (visual / audio / text / unrelated-distractor).
2. **Correct option** must cite a real cue from `seed_event.cues`.
3. **D should be a plausible-sounding background detail** that does NOT actually carry the emotion signal — a typical shortcut answer.
4. **`memory_components` = `["M1"]`**.
5. **Output strict JSON only**.
