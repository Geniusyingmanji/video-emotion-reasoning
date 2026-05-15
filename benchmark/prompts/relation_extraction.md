# Event Relation Extraction Prompt — Stage 3

## Role
You are constructing the **per-episode event DAG** (M2). Given a list of events from Stage 2, infer the causal/temporal/intentional edges among them.

## Edge types
- `temporal_after`: event B happens after A in story time (weak; only label if no stronger relation applies)
- `causal`: A directly causes B; without A, B would not happen
- `emotion_trigger`: A is the proximate emotional trigger for B (an affect_event)
- `counter_action`: B is a deliberate counter to A
- `reveals_belief`: A causes the model to update a belief about a character (e.g., "Walter knows X")
- `reveals_intention`: A surfaces a character's intention behind a later action B
- `predicts_action`: an `affect_event` A makes a future `action_event` B much more likely (this is the T7/T8 ground truth source)

## Hard constraints
- **DAG**: no cycles.
- **Temporal monotonic**: edge A → B implies A.time_span[0] ≤ B.time_span[0].
- **High edge bar**: prefer fewer well-justified edges; only include a `temporal_after` edge if no stronger relation holds AND the events are within 5 minutes of each other (otherwise the relation is too weak).
- **`predicts_action` requires evidence**: cite which cue or summary in A motivates the prediction.

## Output format
Strict JSON. Each edge has `src`, `dst`, `type`, `confidence` ∈ [0,1], `evidence` (free text 1-2 sentences):

```json
{
  "edges": [
    {
      "src": "ep01_ev03",
      "dst": "ep01_ev07",
      "type": "emotion_trigger",
      "confidence": 0.9,
      "evidence": "Hank's call about the meth bust directly triggers Walter's guilt onset in ev07."
    }
  ],
  "_notes": "optional"
}
```

Output JSON only. No prose around it.
