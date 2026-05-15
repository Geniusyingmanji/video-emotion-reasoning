# M7 World / Plot-State Distillation

## Role
Maintain the **propositional plot-state** as facts that hold across the season, with the timestamp at which they become true (or stop being true).

## Input
- All season events
- Cross-episode edges
- M5 personas, M6 relationships

## Output
Strict JSON:

```json
{
  "season": "breaking_bad_s01",
  "facts": [
    {
      "fact_id": "fact_001",
      "statement": "Walter is diagnosed with terminal cancer",
      "becomes_true_at": {"episode": "ep01", "event_id": "ep01_ev03A"},
      "stops_being_true_at": null,
      "evidence_refs": ["ep01_ev03A"],
      "category": "medical | criminal | familial | financial | identity | knowledge"
    },
    {
      "fact_id": "fact_002",
      "statement": "Walter has started cooking methamphetamine",
      "becomes_true_at": {"episode": "ep01", "event_id": "ep01_ev15A"},
      "stops_being_true_at": null,
      "evidence_refs": ["ep01_ev15A", "ep02_ev04A"],
      "category": "criminal"
    },
    {
      "fact_id": "fact_007",
      "statement": "Skyler does NOT know Walter is cooking meth (until end of S1)",
      "becomes_true_at": {"episode": "ep01"},
      "stops_being_true_at": null,
      "evidence_refs": [...],
      "category": "knowledge"
    }
  ],
  "n_facts": ...
}
```

## Hard rules
1. **Every fact must have ≥1 evidence_ref**.
2. **Knowledge facts** (X knows / does not know Y) are crucial — include them; they enable false-belief / ToM questions.
3. **Use canonical character names**.
4. **Output strict JSON only**.
