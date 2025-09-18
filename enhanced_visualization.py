"""
Enhanced visualization tools for CFD data, including streamlines and isotherms.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

def create_streamline_plot(u, v, title="Streamlines"):
    """Generates a streamline plot."""
    ny, nx = u.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.streamplot(x, y, u, v, color='black', linewidth=1, density=2)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    return fig

def create_isotherm_plot(t, title="Isotherms"):
    """Generates an isotherm plot (contour plot of temperature)."""
    ny, nx = t.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    fig, ax = plt.subplots(figsize=(6, 6))
    contour = ax.contour(x, y, t, levels=20, cmap='seismic')
    fig.colorbar(contour, ax=ax, label='Temperature')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    return fig

def create_combined_flow_visualization(u, v, t, title="Flow Visualization"):
    """
    Creates a combined plot with temperature contour background, isotherms, and streamlines.
    """
    ny_t, nx_t = t.shape
    ny_u, nx_u = u.shape
    
    x_t = np.linspace(0, 1, nx_t)
    y_t = np.linspace(0, 1, ny_t)
    
    x_u = np.linspace(0, 1, nx_u)
    y_u = np.linspace(0, 1, ny_u)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Temperature background
    im = ax.imshow(t.T, extent=[0, 1, 0, 1], origin='lower', cmap='seismic', alpha=0.7)
    fig.colorbar(im, ax=ax, label='Temperature')
    
    # Isotherms (contour lines)
    ax.contour(x_t, y_t, t.T, levels=15, colors='white', linewidths=0.7)
    
    # Streamlines
    ax.streamplot(x_u, y_u, u.T, v.T, color='black', linewidth=1, density=1.5)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    
    return fig

def create_enhanced_field_comparison_at_timestep(target_fields, pred_fields, field_names, timestep_for_title, output_path, timestep_for_indexing=0):
    """
    Creates an enhanced side-by-side comparison including individual fields and flow visualizations.
    """
    num_fields = len(field_names)
    u_idx, v_idx, t_idx = -1, -1, -1
    for i, name in enumerate(field_names):
        if name == 'U': u_idx = i
        if name == 'V': v_idx = i
        if name == 'T': t_idx = i

    has_flow_fields = u_idx != -1 and v_idx != -1 and t_idx != -1

    rows = num_fields + 1 if has_flow_fields else num_fields
    
    fig = plt.figure(figsize=(20, 6 * rows))
    gs = gridspec.GridSpec(rows + 1, 3, figure=fig, height_ratios=[0.5] + [5]*rows)

    title_ax = plt.subplot(gs[0, :])
    title_ax.text(0.5, 0.5, f'Enhanced Field Analysis (Timestep: {timestep_for_title})', ha='center', va='center', fontsize=20)
    title_ax.axis('off')

    for i in range(num_fields):
        field_name = field_names[i]
        # Correct slicing: select the right timestep from the first axis
        target = target_fields[i][timestep_for_indexing, :, :]
        pred = pred_fields[i][timestep_for_indexing, :, :]
        error = target - pred
        
        vmin = min(target.min(), pred.min())
        vmax = max(target.max(), pred.max())
        
        # Target
        ax1 = plt.subplot(gs[i + 1, 0])
        im1 = ax1.imshow(target, origin='lower', cmap='seismic', vmin=vmin, vmax=vmax)
        ax1.set_title(f'Target - {field_name}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        fig.colorbar(im1, ax=ax1)
        
        # Prediction
        ax2 = plt.subplot(gs[i + 1, 1])
        im2 = ax2.imshow(pred, origin='lower', cmap='seismic', vmin=vmin, vmax=vmax)
        ax2.set_title(f'Prediction - {field_name}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        fig.colorbar(im2, ax=ax2)
        
        # Error
        ax3 = plt.subplot(gs[i + 1, 2])
        im3 = ax3.imshow(error, origin='lower', cmap='seismic')
        ax3.set_title(f'Error - {field_name}')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        fig.colorbar(im3, ax=ax3)

    if has_flow_fields:
        # Target Flow
        ax_flow1 = plt.subplot(gs[num_fields + 1, 0])
        create_combined_flow_visualization_on_ax(ax_flow1, target_fields[u_idx][timestep_for_indexing, :, :], target_fields[v_idx][timestep_for_indexing, :, :], target_fields[t_idx][timestep_for_indexing, :, :], "Target Flow")
        
        # Predicted Flow
        ax_flow2 = plt.subplot(gs[num_fields + 1, 1])
        create_combined_flow_visualization_on_ax(ax_flow2, pred_fields[u_idx][timestep_for_indexing, :, :], pred_fields[v_idx][timestep_for_indexing, :, :], pred_fields[t_idx][timestep_for_indexing, :, :], "Predicted Flow")
        
        ax_flow3 = plt.subplot(gs[num_fields + 1, 2])
        ax_flow3.axis('off') # Empty plot for alignment

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved enhanced comparison plot to {output_path}")

def create_combined_flow_visualization_on_ax(ax, u, v, t, title):
    """ Helper to draw combined viz on a given matplotlib axis """
    ny_t, nx_t = t.shape
    ny_u, nx_u = u.shape
    x_t, y_t = np.linspace(0, 1, nx_t), np.linspace(0, 1, ny_t)
    x_u, y_u = np.linspace(0, 1, nx_u), np.linspace(0, 1, ny_u)
    
    ax.imshow(t, extent=[0, 1, 0, 1], origin='lower', cmap='seismic', alpha=0.7)
    ax.contour(x_t, y_t, t, levels=15, colors='white', linewidths=0.7)
    ax.streamplot(x_u, y_u, u, v, color='black', linewidth=1, density=1.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')

def create_flow_evolution_animation(u_data, v_data, t_data, save_path, fps=10, title_prefix=""):
    """
    Creates an animation of flow evolution (streamlines + isotherms).
    u_data, v_data, t_data should be of shape (time, height, width).
    """
    num_frames = u_data.shape[0]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    def update(frame):
        ax.clear()
        u, v, t = u_data[frame], v_data[frame], t_data[frame]
        create_combined_flow_visualization_on_ax(ax, u, v, t, f'{title_prefix} - Timestep {frame}')
        
    anim = FuncAnimation(fig, update, frames=num_frames, blit=False)
    
    print(f"Saving animation to {save_path}...")
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close(fig)
    print("Animation saved successfully.")
    return save_path
