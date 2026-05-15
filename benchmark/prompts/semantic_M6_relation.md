# M6 Relationship Network Distillation

## Role
You are building the **typed relationship network** across all named characters in the season.

## Input
- List of all named characters in the season
- All cross-episode events (with participants)
- Cross-episode edge list M3

## Output
Strict JSON:

```json
{
  "season": "breaking_bad_s01",
  "characters": ["Walter", "Skyler", "Jesse", "Hank", "Walt Jr."],
  "edges": [
    {
      "src": "Walter",
      "dst": "Jesse",
      "relation_type": "mentor_to_protege | accomplice | adversary | family | romantic | colleague | stranger",
      "directionality": "directed | mutual",
      "intensity": 0.0-1.0,
      "evolution": [
        {"episode": "ep01", "phase": "first_contact", "evidence_refs": ["ep01_ev05A"]},
        {"episode": "ep03", "phase": "intensified", "evidence_refs": [...]},
        ...
      ],
      "evidence_refs": ["..."]
    }
  ],
  "n_edges": ...
}
```

## Hard rules
1. **Only include relationships with ≥2 evidence_refs**.
2. **Use canonical character names** consistent with M5.
3. **`evolution`** captures relationship phase changes; each phase must cite an event.
4. **Output strict JSON only**.
