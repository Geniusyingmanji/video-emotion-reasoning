# M5 Persona Distillation — OCEAN Five-Factor + Backstory

## Role
You are distilling a **character persona** from observed events across a TV season. Output the **OCEAN Big-Five** five-factor profile, key backstory facts, signature behaviors, and a short narrative summary.

## Input
- The character's name
- The complete cumulative emotion trajectory M4_<character>.json (all waypoints across the season)
- All season events (ep01-epN) involving this character
- The high-confidence cross-episode edge list M3 connecting their events

## Output
Strict JSON with this schema:

```json
{
  "character": "Walter",
  "ocean": {
    "openness":          {"score": 0.55, "evidence_refs": ["ep01_ev03A", "ep03_ev12A", "..."]},
    "conscientiousness": {"score": 0.80, "evidence_refs": ["ep01_ev01A", "ep02_ev07A"]},
    "extraversion":      {"score": 0.30, "evidence_refs": [...]},
    "agreeableness":     {"score": 0.45, "evidence_refs": [...]},
    "neuroticism":       {"score": 0.65, "evidence_refs": [...]}
  },
  "signature_behaviors": [
    {"description": "Avoids confrontation by deflecting to scientific facts", "evidence_refs": [...]},
    ...
  ],
  "backstory_facts": [
    {"fact": "Has been diagnosed with cancer (revealed in S1E1)", "evidence_refs": ["ep01_ev03A"]},
    ...
  ],
  "narrative_summary": "≤120 words, present tense, no spoilers beyond observed events"
}
```

## Hard rules
1. **Every score must be in [0.0, 1.0]**.
2. **Every score must have ≥3 `evidence_refs`**.
3. **No invented facts**: every backstory fact must trace to a specific event.
4. **`evidence_refs` use event_id values** that exist in the supplied event list.
5. **Output strict JSON only**, no prose around it.
6. If you cannot find ≥3 evidence refs for a dimension, set score to `null` and put an explanation in `evidence_refs[0]` like "insufficient evidence: only ev03 supports this".
