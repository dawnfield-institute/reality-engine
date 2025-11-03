# Output Directory Structure

Reality Engine uses timestamped output directories to organize simulation results.

## Directory Layout

```
reality-engine/
├── output/
│   ├── 20251103_143022/          # Timestamped run folder
│   │   ├── run_info.json         # Run metadata and config
│   │   ├── reality_engine_results.pkl  # Saved engine state
│   │   ├── energy_field.png      # Energy field visualization
│   │   ├── info_field.png        # Info field visualization
│   │   ├── memory_field.png      # Memory field visualization
│   │   └── combined_fields.png   # All fields together
│   ├── 20251103_150415/          # Another run
│   │   └── ...
│   └── README.md                 # This stays in git
└── logs/
    ├── reality_engine_20251103_143022.log   # Human-readable log
    ├── reality_engine_20251103_143022.json  # Machine-readable log
    └── README.md
```

## Timestamped Folders

Each simulation run creates a unique timestamped directory: `output/YYYYMMDD_HHMMSS/`

This ensures:
- ✅ No overwriting previous runs
- ✅ Easy chronological organization
- ✅ Clear correspondence between logs and outputs
- ✅ Self-documenting with `run_info.json`

## Run Info JSON

Each run folder contains `run_info.json`:

```json
{
  "timestamp": "20251103_143022",
  "config": {
    "shape": [64, 64, 64],
    "big_bang_perturbation": 1e-10,
    "evolution_steps": 10000,
    "check_interval": 500
  },
  "total_steps": 10000,
  "total_events": 3,
  "total_warnings": 0,
  "log_file": "logs/reality_engine_20251103_143022.log",
  "json_file": "logs/reality_engine_20251103_143022.json",
  "output_dir": "output/20251103_143022"
}
```

## Usage

### Automatic (Recommended)

The `RealityLogger` automatically creates timestamped output directories:

```python
from utils.logger import RealityLogger

# Logger creates output/YYYYMMDD_HHMMSS/ automatically
logger = RealityLogger(experiment_name="my_experiment")

# Use logger's output directory
visualize_all_fields(engine.field, output_dir=str(logger.output_dir))
engine.save_state(str(logger.output_dir / "results.pkl"))

logger.close()  # Writes run_info.json
```

### Examples

Both example scripts use timestamped output:

```bash
# Creates output/YYYYMMDD_HHMMSS/ with full results
python examples/example_run.py

# Creates output/YYYYMMDD_HHMMSS/ with CMB visualizations
python examples/generate_cmb.py
```

After running, check the latest directory:

```bash
# Windows PowerShell
ls output/ | Sort-Object -Descending | Select-Object -First 1

# Linux/Mac
ls -t output/ | head -1
```

## Output Files

### Visualizations

- **`energy_field.png`** - Energy field 2D slice showing 1/r potentials from gravity
- **`info_field.png`** - Information field 2D slice (potential landscape)
- **`memory_field.png`** - Memory field 2D slice (particles and mass concentrations)
- **`combined_fields.png`** - All three fields side-by-side for comparison

### Data Files

- **`reality_engine_results.pkl`** - Pickled engine state for later analysis
  - Full field tensors (E, I, M)
  - Evolution history
  - Emergence report
  - Can be loaded with: `engine = RealityEngine.load_state("path/to/results.pkl")`

### Metadata

- **`run_info.json`** - Configuration, timing, and result summary
  - Links to corresponding log files
  - Event and warning counts
  - Configuration parameters

## Git Behavior

The `.gitignore` is configured to:

- ✅ Keep: `output/README.md`
- ❌ Ignore: `output/*/` (all timestamped directories)
- ✅ Keep: `logs/README.md`
- ❌ Ignore: `logs/*.log` and `logs/*.json`

This prevents committing large output files while preserving documentation.

## Finding Recent Runs

### Latest Run

```bash
# PowerShell
Get-ChildItem output/ | Sort-Object Name -Descending | Select-Object -First 1

# Bash
ls output/ | sort -r | head -1
```

### Runs from Today

```bash
# PowerShell
$today = Get-Date -Format "yyyyMMdd"
Get-ChildItem output/ | Where-Object {$_.Name -like "$today*"}

# Bash
today=$(date +%Y%m%d)
ls output/ | grep "^$today"
```

## Cleanup Old Runs

```bash
# Keep only last 10 runs (PowerShell)
Get-ChildItem output/ | Sort-Object Name -Descending | Select-Object -Skip 10 | Remove-Item -Recurse -Force

# Delete runs older than 7 days (Bash)
find output/ -maxdepth 1 -type d -mtime +7 -exec rm -rf {} \;
```

## Workflow Example

```python
from engine import RealityEngine
from visualize_fields import visualize_all_fields
from utils.logger import RealityLogger

# 1. Create logger (creates timestamped directory)
logger = RealityLogger(experiment_name="my_experiment")

# 2. Run simulation
logger.log_phase("INITIALIZATION", "Creating universe")
engine = RealityEngine(shape=(64, 64, 64))
engine.big_bang(seed_perturbation=1e-10)

logger.log_phase("EVOLUTION", "Evolving for 10000 steps")
report = engine.evolve(steps=10000, check_interval=500, verbose=True)

# 3. Save outputs to timestamped directory
logger.log_phase("VISUALIZATION", "Creating images")
visualize_all_fields(engine.field, output_dir=str(logger.output_dir))

output_file = logger.output_dir / "my_results.pkl"
engine.save_state(str(output_file))

# 4. Close logger (writes run_info.json)
logger.log_phase("COMPLETE", f"Results in {logger.output_dir}/")
logger.close()

print(f"\n✅ All outputs saved to: {logger.output_dir}")
```

This creates:
```
output/20251103_143022/
├── run_info.json
├── my_results.pkl
├── energy_field.png
├── info_field.png
├── memory_field.png
└── combined_fields.png
```

With corresponding logs in:
```
logs/
├── my_experiment_20251103_143022.log
└── my_experiment_20251103_143022.json
```
