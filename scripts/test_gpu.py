"""Quick GPU test to verify CUDA is actually being used."""
import torch
import time

print("Testing GPU utilization...")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Create large tensors on GPU
print("\nAllocating GPU memory...")
size = (512, 512, 512)
a = torch.rand(*size, device=device)
b = torch.rand(*size, device=device)
print(f"Allocated {a.element_size() * a.nelement() * 2 / 1e9:.2f} GB on GPU")

# Force GPU computation
print("\nRunning GPU computation (10 iterations)...")
torch.cuda.synchronize()  # Start fresh
start = time.time()

for i in range(10):
    c = a + b
    c = torch.sin(c)
    c = c * 2.0
    c = torch.roll(c, 1, 0)
    
torch.cuda.synchronize()  # Wait for GPU to finish
elapsed = time.time() - start

print(f"Completed in {elapsed:.3f} seconds")
print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

print("\nIf you see memory usage in Task Manager GPU section, CUDA is working!")
print("If GPU utilization is 0% during this test, there's a driver/CUDA issue.")
