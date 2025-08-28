import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import matplotlib.patches as patches

def save_loss_plot(policy_losses, value_losses, aux_losses=None, filename="loss_curves.png"):
    """
    Saves plots of policy loss, value loss, and optional auxiliary loss curves.
    [PPG Modification] Added optional support for auxiliary loss.
    """
    num_plots = 2 + (1 if aux_losses is not None and len(aux_losses) > 0 else 0)
    plt.figure(figsize=(6 * num_plots, 5))

    # Policy loss subplot
    plt.subplot(1, num_plots, 1)
    plt.plot(policy_losses, label='Policy Loss')
    plt.title('Policy Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Value loss subplot
    plt.subplot(1, num_plots, 2)
    plt.plot(value_losses, label='Value Loss', color='orange')
    plt.title('Value Loss')
    plt.xlabel('Training Step')
    plt.grid(True)
    plt.legend()

    # Auxiliary loss subplot (if provided)
    if num_plots == 3:
        plt.subplot(1, num_plots, 3)
        plt.plot(aux_losses, label='Auxiliary Loss', color='green')
        plt.title('Auxiliary Loss')
        plt.xlabel('Training Step')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def calculate_total_reward_from_log(log_filename):
    """
    [Added] Calculates total reward from a specified episode log file.
    """
    try:
        if not os.path.exists(log_filename):
            print(f"⚠️ Log file not found for reward calculation: {log_filename}")
            return 0.0
        
        data = pd.read_csv(log_filename)
        
        if 'reward' in data.columns and not data['reward'].empty:
            return data['reward'].sum()
        else:
            print(f"⚠️ 'reward' column not found or empty in {log_filename}")
            return 0.0
            
    except Exception as e:
        print(f"❌ Failed to calculate total reward from {log_filename}: {e}")
        return 0.0


def plot_env_cumulative_rewards(env_id, rewards, output_prefix="cumulative_rewards"):
    """
    [Modified] Plots cumulative reward curve over episodes for a single environment.
    Saves file in project root directory with environment ID in filename.
    """
    if not rewards:
        return None

    try:
        plt.figure(figsize=(10, 6))
        episodes = range(1, len(rewards) + 1)
        plt.plot(episodes, rewards, marker='o', linestyle='-', label=f'Env {env_id} Total Reward')

        plt.title(f'Total Reward per Episode for {env_id}')
        plt.xlabel('Episode Number')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.legend()
        
        # Generate environment-specific filename for root directory
        output_filename = f"{output_prefix}_{env_id}.png"
        
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()

        # Return generated filename for logging purposes
        return output_filename
    except Exception as e:
        print(f"❌ Failed to plot cumulative rewards for {env_id}: {e}")
        return None


def plot_episode_data(log_filename, output_filename, reward_pos=None, obstacle_pos=None):
    """
    Reads episode log data and generates analysis charts including trajectory,
    distance to reward, velocity projection, power metrics, angle information,
    and reward history.

    Args:
        log_filename (str): Path to source log file.
        output_filename (str): Path to save generated chart.
        reward_pos (list or np.array): [x, y] coordinates of reward point.
        obstacle_pos (list or np.array): [x, y] coordinates of obstacle.
    """
    print(f"🔍 Starting plot generation: {log_filename} -> {output_filename}")
    
    try:
        if not os.path.exists(log_filename):
            print(f"❌ Log file does not exist: {log_filename}")
            return
            
        if os.path.getsize(log_filename) == 0:
            print(f"❌ Log file is empty: {log_filename}")
            return
            
        print(f"📁 File exists, size: {os.path.getsize(log_filename)} bytes")

        # Load data from log file
        data = np.loadtxt(log_filename, delimiter=',', skiprows=1)
        print(f"📊 Data loaded successfully, shape: {data.shape}")

        # Reshape 1D array to 2D for consistency if only one row exists
        if data.ndim == 1:
            data = data.reshape(1, -1)
            print(f"📊 Reshaped data shape: {data.shape}")

        # Remove first row if time regression detected (indicating new episode start)
        if data.shape[0] >= 2 and data[0, 0] > data[1, 0]:
            print(f"⚠️ Time regression detected, removing first row")
            data = data[1:, :]
            print(f"📊 Processed data shape: {data.shape}")

        # Validate data column count
        expected_cols = 16  # time,fish_x,fish_y,vel_x,vel_y,dist_to_reward,angle_to_reward,vel_projection,constraint_power,inertia_power,total_power,reward_metric,reward,cos,sin,rot_momentum_z
        if data.shape[1] < expected_cols:
            print(f"❌ Insufficient data columns: expected {expected_cols}, got {data.shape[1]}")
            print(f"📄 First few data rows:")
            print(data[:min(3, data.shape[0]), :])
            return

        print(f"✅ Data validation passed, starting plot generation...")

        # Create 2x3 subplot layout
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        fig.suptitle(f'Episode Analysis Report: {os.path.basename(log_filename)}', fontsize=16)

        # 1. Fish trajectory plot
        ax = axes[0, 0]
        ax.plot(data[:, 1], data[:, 2], label='Trajectory', marker='.', markersize=2, linestyle='-')
        ax.scatter(data[0, 1], data[0, 2], c='green', s=100, label='Start', zorder=5)
        ax.scatter(data[-1, 1], data[-1, 2], c='red', s=100, label='End', zorder=5)
        if reward_pos is not None:
            ax.scatter(reward_pos[0], reward_pos[1], c='blue', s=200, marker='*', label='Reward', zorder=5)
        
        if obstacle_pos is not None:
            cx, cy = obstacle_pos
            radius = 0.5 / 2.0  # Diameter 0.25, radius 0.125
            
            # Draw semicircle (90° to 270°)
            arc = patches.Arc((cx, cy), width=radius*2, height=radius*2, angle=0,
                              theta1=90, theta2=270, color='purple', linewidth=2, label='Obstacle')
            ax.add_patch(arc)
            
            # Draw vertical line segment
            ax.plot([cx, cx], [cy - radius, cy + radius], color='purple', linewidth=2)
        
        ax.set_xlim([-7.6, 0.0])
        ax.set_ylim([-4.0, 4.0])
        ax.set_title('Fish Body Trajectory')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')

        # 2. Distance to reward over time
        ax = axes[0, 1]
        ax.plot(data[:, 0], data[:, 5], label='Distance to Reward')  # Column 5: dist_to_reward
        ax.set_title('Distance to Reward vs. Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance')
        ax.legend()
        ax.grid(True)

        # 3. Velocity projection over time
        ax = axes[0, 2]
        ax.plot(data[:, 0], data[:, 7], label='Velocity Projection')  # Column 7: vel_projection
        ax.set_title('Velocity Projection on Reward Direction vs. Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Projected Velocity')
        ax.legend()
        ax.grid(True)

        # 4. Power metrics over time
        ax = axes[1, 0]
        ax.plot(data[:, 0], data[:, 8], label='Constraint Power')  # Column 8: constraint_power
        ax.plot(data[:, 0], data[:, 9], label='Inertia Power')     # Column 9: inertia_power
        ax.plot(data[:, 0], data[:, 10], label='Total Power', linestyle='--', color='black')  # Column 10: total_power
        ax.set_title('Power vs. Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power')
        ax.legend()
        ax.grid(True)
        ax.set_ylim(bottom=0)  # Set minimum y-axis value to 0

        # 5. Angle (cos/sin) and rotational momentum over time
        ax = axes[1, 1]
        ax.plot(data[:, 0], data[:, 13], label='cos(Angle to Reward)')  # Column 13: cos_angle
        ax.plot(data[:, 0], data[:, 14], label='sin(Angle to Reward)')  # Column 14: sin_angle
        ax.plot(data[:, 0], data[:, 15], label='Rot. Momentum Z * 1000', linestyle='--')  # Rotational inertia
        ax.set_title('Angle & Rot. Momentum vs. Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

        # 6. Reward history with total reward annotation
        ax = axes[1, 2]
        reward_history = data[:, 12]  # Column 13: reward values
        total_reward = np.sum(reward_history)

        ax.plot(reward_history, label='Reward per Step', color='purple', marker='o', markersize=2, linestyle='-')
        ax.set_title('Reward History')
        ax.set_xlabel('Step Index')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True)
        
        # Annotate total reward on plot
        ax.text(0.95, 0.95, f'Total Reward: {total_reward:.2f}',
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes,
                color='blue', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 Created output directory: {output_dir}")
        
        plt.savefig(output_filename, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✅ Episode analysis plot saved to: {output_filename}")

    except Exception as e:
        print(f"❌ Failed to plot episode analysis: {e}")
        import traceback
        traceback.print_exc()  # Print full error stack
