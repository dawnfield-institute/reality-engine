"""Benchmark GPU performance - run this and watch Task Manager GPU usage."""
import torch
import time
from core.dawn_field import DawnField

print("=" * 60)
print(" GPU PERFORMANCE BENCHMARK")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Create field
print("\n📊 Initializing field (64³)...")
field = DawnField(shape=(64, 64, 64))

print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")
print(f"            {torch.cuda.memory_reserved(0) / 1e9:.2f} GB reserved")

# Warm-up
print("\n🔥 Warming up GPU...")
for _ in range(10):
    field.evolve_step()
torch.cuda.synchronize()

# Benchmark
n_steps = 1000
print(f"\n⚡ Running {n_steps} evolution steps...")
print("   Watch Task Manager -> GPU section for utilization!")
print()

torch.cuda.synchronize()
start = time.time()

for i in range(n_steps):
    field.evolve_step()
    if (i + 1) % 100 == 0:
        print(f"   Step {i+1}/{n_steps}...")

torch.cuda.synchronize()
elapsed = time.time() - start

print(f"\n✅ Completed {n_steps} steps in {elapsed:.2f} seconds")
print(f"   {n_steps / elapsed:.1f} steps/second")
print(f"   {elapsed / n_steps * 1000:.2f} ms/step")

print(f"\n📈 GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")
print(f"              {torch.cuda.memory_reserved(0) / 1e9:.2f} GB reserved")

print("\n" + "=" * 60)
print(" If GPU utilization was >0% during run, GPU is working!")
print(" If it stayed at 0%, operations are falling back to CPU.")
print("=" * 60)
