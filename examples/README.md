# Examples

Example scripts demonstrating Reality Engine capabilities.

## Available Examples

### `example_run.py`
Basic universe simulation with full logging.

```bash
python examples/example_run.py
```

**What it does:**
- Creates a 64³ field
- Triggers Big Bang herniation
- Evolves for 10,000 steps
- Detects emergent phenomena
- Generates comprehensive logs

**Expected runtime:** ~2 minutes on GPU

### `generate_cmb.py`
Generate cosmic microwave background-like visualizations.

```bash
python examples/generate_cmb.py
```

**What it does:**
- Runs full simulation
- Extracts 2D field slices
- Creates CMB-style images
- Saves to `output/` directory

**Expected output:**
- `energy_field.png`
- `info_field.png`
- `memory_field.png`
- `combined_fields.png`

## Customization

Modify parameters in the scripts:

```python
# Change universe size
universe = RealityEngine(shape=(128, 128, 128))

# Adjust evolution duration
report = universe.evolve(steps=20000, check_interval=1000)

# Tune Big Bang strength
universe.big_bang(seed_perturbation=1e-9)
```

## Output

All examples generate:
- Console output with emergence notifications
- Timestamped logs in `logs/`
- Visualizations in `output/` (where applicable)
