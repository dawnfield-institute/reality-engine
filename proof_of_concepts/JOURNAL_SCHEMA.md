# GAIA-Style POC Journal Schema

> Standard format for documenting POC research journeys in Reality Engine

---

## File Naming Convention

```
YYYY-MM-DD_descriptive_slug.md
```

Examples:
- `2026-01-08_initial_substrate_tests.md`
- `2026-01-09_scale_comparison_discovery.md`

---

## Required Sections

Every journal entry MUST include:

### 1. Summary (Top)
Brief 2-3 sentence summary of what was attempted and key outcomes.

### 2. Timeline
Chronological log of activities with status markers:

```markdown
### HH:MM - Activity Type

Description of what was done.

**Status:** ✅ Confirmed | ❌ Failed | 🔄 In Progress | 💡 Insight
```

Activity Types:
- **Setup**: Environment, dependencies, configuration
- **Experiment**: Running actual tests
- **Analysis**: Interpreting results
- **Discovery**: Unexpected findings
- **Bug Fix**: Resolving issues
- **Planning**: Next steps, pivots

### 3. Key Findings
Bullet list of most important discoveries.

### 4. Metrics Collected
Quantitative data gathered (tables preferred).

### 5. Challenges Encountered
What went wrong or was harder than expected.

### 6. Next Steps
Concrete actions for follow-up.

---

## Template

```markdown
# Journal: [Descriptive Title]

**Date:** YYYY-MM-DD  
**POC:** POC-XXX  
**Author:** [Name]  
**Status:** 🔄 In Progress | ✅ Complete | ❌ Blocked

---

## Summary

[2-3 sentences summarizing the session]

---

## Timeline

### HH:MM - Setup

[What was configured]

**Status:** ✅ Confirmed

### HH:MM - Experiment

[What was tested]

**Status:** 🔄 In Progress

---

## Key Findings

- Finding 1
- Finding 2

---

## Metrics Collected

| Metric | Value | Notes |
|--------|-------|-------|
| metric_1 | X | description |

---

## Challenges Encountered

- Challenge 1

---

## Next Steps

- [ ] Task 1
- [ ] Task 2
```
