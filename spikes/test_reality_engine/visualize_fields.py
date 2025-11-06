"""
Field Visualization - Create CMB-like Images

Generates 2D slice visualizations of the 3D fields showing:
- Memory field (particles/mass concentrations)
- Energy field (with radial gradients)
- Information field (potential landscape)

Like the cosmic microwave background, these should show:
- Localized hotspots (particles)
- Radial gradients (gravity potentials)
- Fractal structure (quantum/emergent complexity)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def visualize_field_slice(field, title="Field", cmap="inferno", z_slice=None):
    """
    Visualize 2D slice of 3D field.
    
    Args:
        field: 3D tensor
        title: Plot title
        cmap: Colormap
        z_slice: Z index to slice (default: middle)
    """
    # Move to CPU if needed
    if field.is_cuda:
        field = field.cpu()
    
    # Get middle slice if not specified
    if z_slice is None:
        z_slice = field.shape[2] // 2
    
    # Extract 2D slice
    slice_2d = field[:, :, z_slice].numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot with colorbar
    im = ax.imshow(slice_2d.T, origin='lower', aspect='auto', cmap=cmap)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{title} (XY slice at Z={z_slice})')
    plt.colorbar(im, ax=ax)
    
    return fig


def visualize_all_fields(dawn_field, output_dir='output', z_slice=None):
    """
    Create CMB-like visualizations of all three fields.
    
    Args:
        dawn_field: DawnField instance
        output_dir: Directory to save images
        z_slice: Z index to slice (default: middle)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Memory field (particles/mass)
    fig_m = visualize_field_slice(
        dawn_field.M, 
        "Memory Field (Particles/Mass)",
        cmap='hot',
        z_slice=z_slice
    )
    fig_m.savefig(output_path / 'memory_field.png', dpi=150, bbox_inches='tight')
    plt.close(fig_m)
    print(f"✓ Saved memory_field.png")
    
    # Energy field (should show radial gradients from herniations)
    fig_e = visualize_field_slice(
        dawn_field.E,
        "Energy Field (with 1/r potentials)",
        cmap='plasma',
        z_slice=z_slice
    )
    fig_e.savefig(output_path / 'energy_field.png', dpi=150, bbox_inches='tight')
    plt.close(fig_e)
    print(f"✓ Saved energy_field.png")
    
    # Information field (potential landscape)
    fig_i = visualize_field_slice(
        dawn_field.I,
        "Info Field (Potential)",
        cmap='viridis',
        z_slice=z_slice
    )
    fig_i.savefig(output_path / 'info_field.png', dpi=150, bbox_inches='tight')
    plt.close(fig_i)
    print(f"✓ Saved info_field.png")
    
    # Combined figure (all three side by side)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    if z_slice is None:
        z_slice = dawn_field.M.shape[2] // 2
    
    # Memory
    im1 = axes[0].imshow(dawn_field.M[:, :, z_slice].cpu().numpy().T, 
                         origin='lower', aspect='auto', cmap='hot')
    axes[0].set_title('Memory Field (XY slice at Z={})'.format(z_slice))
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0])
    
    # Energy
    im2 = axes[1].imshow(dawn_field.E[:, :, z_slice].cpu().numpy().T,
                         origin='lower', aspect='auto', cmap='plasma')
    axes[1].set_title('Energy Field (XY slice at Z={})'.format(z_slice))
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1])
    
    # Information
    im3 = axes[2].imshow(dawn_field.I[:, :, z_slice].cpu().numpy().T,
                         origin='lower', aspect='auto', cmap='viridis')
    axes[2].set_title('Info Field (XY slice at Z={})'.format(z_slice))
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    fig.savefig(output_path / 'combined_fields.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved combined_fields.png")
    
    print(f"\n📊 Field visualizations saved to {output_dir}/")


if __name__ == '__main__':
    print("This module is for visualization. Import and use visualize_all_fields().")
