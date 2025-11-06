"""
Generate CMB-like visualization from final Reality Engine state.
"""

import sys
sys.path.insert(0, '.')

from engine import RealityEngine
from visualize_fields import visualize_all_fields
from utils.logger import RealityLogger
from datetime import datetime

# Create logger for timestamped output
logger = RealityLogger(experiment_name="cmb_generation")
logger.log_phase("CMB GENERATION", "Creating CMB-like visualizations with QBE dynamics")

# Create and run simulation
print("🌌 Running QBE-constrained Reality Engine simulation...")
engine = RealityEngine(shape=(64, 64, 64))

# Big Bang
print("💥 Big Bang...")
logger.log_phase("BIG BANG", "Initiating herniation cascade")
engine.big_bang(seed_perturbation=1e-10)

# Evolve
print("⏳ Evolving for 10000 steps with Quantum Balance Equation...")
logger.log_phase("EVOLUTION", "Evolving with pure QBE dynamics")
report = engine.evolve(steps=10000, check_interval=500, verbose=True)

# Visualize
print("📊 Generating CMB-like visualizations...")
logger.log_phase("VISUALIZATION", "Creating field images")
visualize_all_fields(
    engine.field,
    output_dir=str(logger.output_dir)
)

logger.log_validation({
    'PAC Conservation': report.get('pac_conservation', False),
    'Particles Emerged': report.get('particles_emerged', False),
    'Gravity Emerged': report.get('gravity_emerged', False)
})

print(f"\n✅ Complete! Check {logger.output_dir}/ for CMB visualizations.")
print(f"   Particles emerged: {report.get('particles_emerged', False)}")
print(f"   Gravity emerged: {report.get('gravity_emerged', False)}")

logger.close()
