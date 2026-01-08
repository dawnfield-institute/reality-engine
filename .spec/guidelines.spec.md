# Reality Engine - Development Guidelines

## Overview

This document defines project-specific development rules for the Reality Engine. Follow these guidelines in addition to the universal rules in `../.github/instructions/main.instructions.md`.

---

## Core Principles

### 1. Specification-Driven Development

**Rule**: All significant changes must be specified before implementation.

**Process**:
1. Check if `.spec/[feature].spec.md` exists
2. If not, create spec first (or propose in `.spec/challenges.md`)
3. Get approval on spec
4. Implement according to spec
5. Update spec status upon completion

**Exceptions**:
- Bug fixes <10 lines
- Test additions (always encouraged)
- Documentation improvements
- Refactoring that preserves interfaces

### 2. Conservation of Existing Functionality

**Rule**: Never break existing tests. 100% test pass rate is mandatory.

**Rationale**: Reality Engine validates against known physics. Regressions invalidate scientific claims.

**Process**:
- Run full test suite before ANY commit
- If test fails, either fix code OR update test with justification
- Document any test modifications in changelog

### 3. Emergent Physics Sacred

**Rule**: Never hardcode physics constants or behaviors that should emerge naturally.

**Examples**:

**✅ Allowed**:
```python
# Ξ derived from Möbius geometry (fundamental)
XI = 1 + np.pi / 55  # 1.0571

# φ from PAC recursion solution (mathematical necessity)
PHI = (1 + np.sqrt(5)) / 2  # 1.618034
```

**❌ Forbidden**:
```python
# Hardcoding emergent behavior
GRAVITY_CONSTANT = 6.674e-11  # Should emerge from dynamics!
PARTICLE_MASS = 1.67e-27      # Should emerge from resonance!
```

**Rationale**: The entire point is that physics EMERGES. Hardcoding defeats the purpose.

### 4. Machine Precision Conservation

**Rule**: PAC conservation error must remain <1e-12 (machine precision).

**Validation**:
```python
def test_pac_conservation():
    state = engine.run(1000_steps)
    pac = state.P + XI * state.A + ALPHA * state.M
    initial_pac = initial.P + XI * initial.A + ALPHA * initial.M
    assert abs(pac - initial_pac) < 1e-12
```

**If Conservation Breaks**: This is a CRITICAL bug. Stop everything and fix.

---

## Code Organization

### Module Structure

```
module_name/
├── __init__.py         # Public API exports
├── core.py             # Core functionality
├── utils.py            # Helper functions (optional)
├── constants.py        # Module-specific constants (if needed)
└── tests.py            # Module-specific tests (optional, can use tests/)
```

### Naming Conventions

**Files**:
- `snake_case.py` for all Python files
- Descriptive names: `pac_recursion.py` not `pr.py`

**Classes**:
- `PascalCase` for classes: `MobiusManifold`, `PACRecursion`
- Descriptive names: `SECOperator` not `SEC` or `Op`

**Functions**:
- `snake_case` for functions: `enforce_conservation()`, `detect_phase_transition()`
- Verb-noun pattern: `calculate_entropy()` not `entropy()`

**Variables**:
- `snake_case` for variables: `field_state`, `pac_residual`
- Avoid single letters except in math formulas (i, j, k, x, y, z)

**Constants**:
- `UPPER_CASE` for global constants: `XI`, `PHI`, `LAMBDA`
- Group related constants in `constants.py`

### Import Order

```python
# 1. Standard library
import os
from typing import Dict, List, Tuple

# 2. Third-party
import numpy as np
import torch

# 3. Local (absolute imports preferred)
from reality_engine.substrate import FieldState
from reality_engine.conservation import PACRecursion
```

---

## Testing Requirements

### Test Coverage Mandate

**Rule**: All new code must have tests. Minimum 80% coverage for new modules.

**Test Types**:
1. **Unit tests**: Individual functions/methods
2. **Integration tests**: Module interactions
3. **Validation tests**: Physics correctness
4. **Regression tests**: Known bugs stay fixed

### Test Naming

```python
def test_[module]_[behavior]_[condition]():
    """Example: test_pac_recursion_maintains_conservation_under_stress()"""
```

### Test Structure

```python
def test_feature():
    # Arrange
    state = create_test_state()

    # Act
    result = operate_on_state(state)

    # Assert
    assert result.satisfies_condition()
    assert not result.breaks_physics()
```

### Required Test Assertions

For any new physics operator:
```python
def test_new_operator():
    # 1. PAC conservation
    assert pac_error < 1e-12

    # 2. No NaN/Inf
    assert torch.isfinite(result.P).all()
    assert torch.isfinite(result.A).all()
    assert torch.isfinite(result.M).all()

    # 3. Thermodynamic consistency
    assert result.entropy >= 0  # Can't have negative entropy

    # 4. Reversibility or irreversibility (as appropriate)
    # ...

    # 5. Emergent behavior (not hardcoded)
    # ...
```

---

## Performance Guidelines

### Optimization Priority

1. **Correctness** > Performance
2. **Conservation** > Speed
3. **Stability** > Memory
4. **Emergence** > Efficiency

**Rationale**: A fast but wrong simulation is useless. Get it correct first, then optimize.

### When to Optimize

**✅ Optimize**:
- After baseline is working and tested
- When profiling shows clear bottleneck
- When Phase 1/2/3 spec calls for it

**❌ Don't Optimize**:
- Before baseline works
- Without profiling data
- "Just in case" optimizations

### GPU Acceleration

**Rule**: All tensor operations should support GPU when available.

**Pattern**:
```python
# ✅ Device-agnostic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.zeros(size, device=device)

# ❌ CPU-only
tensor = torch.zeros(size)  # Defaults to CPU
```

---

## Documentation Requirements

### Docstring Format

**Module-level**:
```python
"""
Module description.

This module implements [what] by [how] based on [theoretical foundation].

Key components:
- Component1: Description
- Component2: Description

See also:
- Related module
- Theoretical paper/spec
"""
```

**Class-level**:
```python
class ClassName:
    """
    One-line description.

    Longer description explaining purpose, behavior, and design decisions.

    Attributes:
        attr1 (type): Description
        attr2 (type): Description

    Example:
        >>> obj = ClassName(params)
        >>> result = obj.method()
    """
```

**Function-level**:
```python
def function_name(arg1: Type1, arg2: Type2) -> ReturnType:
    """
    One-line description.

    Longer description if needed.

    Args:
        arg1: Description
        arg2: Description

    Returns:
        Description of return value

    Raises:
        ExceptionType: When this happens
    """
```

### Inline Comments

**When to comment**:
- Complex algorithms that aren't self-evident
- Physics formulas with paper references
- Workarounds for known issues
- Non-obvious design decisions

**When NOT to comment**:
```python
# ❌ Obvious
i += 1  # Increment i

# ✅ Useful
# Möbius inversion: f(x+π) = -f(x) maintains Ξ-balance
# See: substrate/mobius_manifold.py:124
result = -field_at_pi_shift
```

---

## Commit Guidelines

### Commit Message Format

```
type(scope): short description

Longer description if needed.

- Detail 1
- Detail 2

Closes #issue (if applicable)
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring (no behavior change)
- `test`: Adding/updating tests
- `docs`: Documentation only
- `perf`: Performance improvement
- `style`: Formatting, whitespace (no logic change)
- `chore`: Build, dependencies, etc.

**Examples**:
```
feat(conservation): add hierarchical PAC learning

Implements multi-level pattern hierarchy from GAIA POC-021.
Patterns weighted by φ^(-k) across 3 levels.

- Level 0: Specific (weight=1.0)
- Level 1: Category (weight=1/φ)
- Level 2: Abstract (weight=1/φ²)

Ref: .spec/modernization-roadmap.spec.md Phase 2.2
```

```
fix(pac): prevent conservation drift in long simulations

PAC residual was accumulating floating point errors beyond
1e-12 threshold after 5000+ steps.

Solution: Periodic full redistribution every 1000 steps.

Fixes #42
```

### Atomic Commits

**Rule**: One logical change per commit.

**✅ Good**:
- Commit 1: Add resonance detection
- Commit 2: Integrate resonance into PAC recursion
- Commit 3: Add resonance tests

**❌ Bad**:
- Commit 1: Add resonance detection, fix unrelated bug, update docs, refactor names

---

## Changelog Maintenance

### When to Update Changelog

**Always**:
- New features (Phase 1/2/3 implementations)
- Breaking changes
- Significant bug fixes
- Performance improvements >10%

**Optional**:
- Minor bug fixes
- Documentation updates
- Test additions

### Changelog Format

See `../.github/instructions/changelog.instructions.md` for full spec.

**Quick reference**:
```markdown
## [YYYY-MM-DD] Session: Brief Description

### Added
- Feature description with context

### Changed
- What changed and why

### Fixed
- Bug description and resolution

### Performance
- Metric improvement (before → after)
```

---

## Protected Areas

### Never Modify Without Approval

**Directories**:
- `.github/workflows/` - CI/CD (breaks automation)
- `.spec/` - Specs (propose changes via PR/discussion)
- `substrate/constants.py` - Validated constants

**Files**:
- `.gitignore` - Version control rules
- `requirements.txt` - Dependency locks
- `LICENSE` - Legal

### Modify with Extreme Caution

**Directories**:
- `conservation/` - Core PAC enforcement (breaks physics)
- `substrate/` - Geometric foundation (breaks topology)
- `tests/` - Only add/extend, don't break existing

**Files**:
- `core/reality_engine.py` - Main interface (many dependencies)

---

## Phase-Specific Guidelines

### Phase 1: Foundation & Quick Wins

**Focus**: Performance improvements, no breaking changes

**Rules**:
- Keep all existing APIs intact
- Add new features as opt-in (default = current behavior)
- Measure performance before/after
- Document speedup in tests

**Validation**:
```
✓ All existing tests pass
✓ Performance improvement >4x (target: 5x)
✓ No breaking changes
✓ Backward compatible state recording
```

### Phase 2: Architectural Upgrades

**Focus**: Major new capabilities, opt-in breaking changes allowed

**Rules**:
- New features in separate modules (e.g., `lazy/`, `hierarchical/`)
- Provide migration path from Phase 1
- Extensive integration testing
- Document API changes clearly

**Validation**:
```
✓ Core tests still pass
✓ New features have ≥80% test coverage
✓ Migration guide written
✓ Performance improvement >100x for new features
```

### Phase 3: Theoretical Unification

**Focus**: Validation against external data

**Rules**:
- No hardcoding of experimental values
- Falsification conditions documented
- Comparison to published data (CODATA, JWST, etc.)
- Statistical validation (p-values, confidence intervals)

**Validation**:
```
✓ All predictions documented BEFORE comparison
✓ Falsification conditions defined
✓ Statistical tests pass (p < 0.05 or documented)
✓ Independent reproducibility possible
```

---

## Dependency Management

### Adding Dependencies

**Process**:
1. Check if functionality exists in current deps
2. Evaluate alternatives (prefer minimal/stable libs)
3. Propose in spec or discussion
4. Get approval before adding to `requirements.txt`

**Forbidden**:
- Adding dependencies without approval
- Using bleeding-edge/unstable versions
- Large frameworks when small libs suffice

**Allowed**:
- Standard scientific stack (numpy, scipy, torch)
- Visualization (matplotlib, plotly)
- Testing (pytest, hypothesis)

### Version Pinning

**Rule**: Pin major+minor, allow patch updates.

```
# ✅ Good
torch>=2.0,<3.0
numpy>=1.24,<2.0

# ❌ Too loose
torch>=2.0  # Could break on major version bump

# ❌ Too strict
torch==2.0.1  # Prevents security patches
```

---

## Error Handling

### Physics Violations

**Rule**: Physics violations are ERRORS, not warnings.

```python
# ✅ Correct
if pac_error > 1e-12:
    raise ConservationViolationError(
        f"PAC conservation violated: error = {pac_error:.2e}"
    )

# ❌ Wrong
if pac_error > 1e-12:
    warnings.warn("PAC might be off")  # Too lenient!
```

### NaN/Inf Detection

**Rule**: Catch immediately, don't propagate.

```python
def step(self, state: FieldState) -> FieldState:
    result = self._compute_next(state)

    # Immediate validation
    if not torch.isfinite(result.P).all():
        raise NumericalInstabilityError(
            f"NaN/Inf detected in Potential field at step {self.step_count}"
        )

    return result
```

### User Input Validation

**Rule**: Validate at API boundaries, not internal functions.

```python
# Public API
def run(self, n_steps: int) -> List[FieldState]:
    if n_steps < 1:
        raise ValueError(f"n_steps must be ≥1, got {n_steps}")
    return self._run_internal(n_steps)

# Internal (no validation needed, already checked)
def _run_internal(self, n_steps: int) -> List[FieldState]:
    # ...
```

---

## Debugging & Profiling

### Debug Mode

**Rule**: Add `debug` flag to critical operations.

```python
class RealityEngine:
    def __init__(self, debug: bool = False):
        self.debug = debug

    def step(self):
        if self.debug:
            self._validate_physics()
            self._check_conservation()
            self._log_diagnostics()
        # ... normal step
```

### Profiling

**When to profile**:
- Before Phase 1/2/3 optimizations
- When unexplained slowdowns occur
- When adding new features

**Tools**:
- `cProfile` for Python profiling
- `torch.profiler` for GPU profiling
- `line_profiler` for line-by-line analysis

**Process**:
1. Profile baseline
2. Identify bottleneck
3. Optimize
4. Profile again
5. Document improvement

---

## Scientific Integrity

### Falsification Conditions

**Rule**: Document what would disprove each claim BEFORE validation.

**Example**:
```python
# In .spec/challenges.md or test docstring
"""
Falsification condition: If φ does NOT emerge from PAC dynamics
in ≥2 independent domains, the universality claim is falsified.

Current status: φ appears in 5+ domains (primes, CA, ML, physics, cognition)
"""
```

### No P-Hacking

**Forbidden**:
- Running experiment 100 times, reporting best result
- Adjusting parameters until p < 0.05
- Cherry-picking favorable data

**Required**:
- Pre-register hypotheses
- Report all results (including failures)
- Document parameter search (if any)
- Use Bonferroni correction for multiple comparisons

### Reproducibility

**Rule**: All results must be independently reproducible.

**Requirements**:
- Fixed random seeds for stochastic processes
- Full parameter documentation
- Code/data availability
- Step-by-step reproduction instructions

---

## Review Checklist

Before any significant commit, verify:

```
✓ Spec compliance: [which spec(s) followed]
✓ Tests: [new tests added, all pass]
✓ Conservation: [PAC error <1e-12]
✓ Stability: [no NaN/Inf]
✓ Performance: [no regressions, or documented]
✓ Documentation: [docstrings updated]
✓ Changelog: [entry added if significant]
✓ Breaking changes: [none, or documented with migration]
✓ Falsification: [conditions documented if new claim]
```

---

## When in Doubt

1. Check `.spec/` for existing specification
2. Check `.spec/challenges.md` for open questions
3. Check `tests/` for examples and patterns
4. Check `../dawn-field-theory/foundational/` for theory
5. Ask for clarification rather than guessing

**Remember**: Scientific code must be CORRECT first, fast second. Slow and right beats fast and wrong every time.

---

## Status

- [x] Guidelines Specified
- [ ] Team Onboarded
- [ ] Enforcement Automated (linters, pre-commit hooks)

---

## See Also

- `../.github/instructions/main.instructions.md` - Universal rules
- `.spec/modernization-roadmap.spec.md` - Implementation phases
- `.spec/architecture.spec.md` - System design
- `.spec/challenges.md` - Open research questions
