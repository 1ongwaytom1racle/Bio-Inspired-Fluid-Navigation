# -*- coding: utf-8 -*-

# model_server.py - Multi-environment server with state recording support
import socket
import json
import time
import signal
import sys
import os
import argparse
import threading
from collections import defaultdict, deque
import random
import numpy as np
import torch

from state_processor import StateProcessor
# [PPG modification] Import PPGAgent (Note: PPG has degenerated to PPO due to aux_loss=0)
from ppg_agent import PPGAgent, ACTION_MAP, ACTION_DIM
# Import plotting utilities
from plot_utils import save_loss_plot, plot_episode_data, calculate_total_reward_from_log, plot_env_cumulative_rewards

TOTAL_TRAJECTORY_LENGTH_TO_UPDATE = 200  # Trigger update when total trajectory length reaches this value

# Training stage management
TRAINING_STAGE = 1  # Initial stage is 1
STAGE_EVALUATION_WINDOW = 10  # Use recent 20 episodes for evaluation
STAGE_UPGRADE_THRESHOLD = 0.6  # Enter next stage when success rate reaches 70%
# Store success/failure records for each environment's recent N episodes
stage_eval_history = defaultdict(lambda: deque(maxlen=STAGE_EVALUATION_WINDOW))

# State processor configuration
PRESSURE_DIM = 15  # Note: Pressure data is no longer used in actual state processing
ACTION_HISTORY_LENGTH = 2
HISTORY_LENGTH = 20  # Configurable history length parameter

# Feature dimensions for each feature stream
FEATURE_DIMS = {
    'visual': 32,     # Visual feature embedding dimension
    'pressure': 16,   # Pressure history GRU hidden layer and embedding dimension (unused)
    'action': 32      # Action history + velocity feature embedding dimension
}

# Store dynamic reward positions for each environment
env_reward_positions = {}

def generate_new_reward_pos(center=(-4.0, 0.0), inner_radius=2.5, outer_radius=3.0):
    """
    Generate random reward position in annular region around center.
    Adjusts angle range based on global TRAINING_STAGE.
    """
    global TRAINING_STAGE
    
    # Uniform sampling on radius squared for uniform area distribution
    r_sq = np.random.uniform(inner_radius**2, outer_radius**2)
    r = np.sqrt(r_sq)
    
    # Determine angle range based on training stage
    if TRAINING_STAGE == 1:
        # Stage 1: ±30° to ±60° forward region, learn small angle turns
        angle_range_outer = np.pi / 3.0  # 60 degrees
        angle_range_inner = np.pi / 6.0  # 30 degrees
        # Randomly choose left or right region
        if np.random.rand() < 0.5:
            # Left region
            theta = np.random.uniform(np.pi - angle_range_outer, np.pi - angle_range_inner)
        else:
            # Right region
            theta = np.random.uniform(np.pi + angle_range_inner, np.pi + angle_range_outer)

    elif TRAINING_STAGE == 2:
        # Stage 2: ±60° to ±120° forward region, learn large angle turns
        angle_range_outer = 2 * np.pi / 3.0 # 120 degrees
        angle_range_inner = np.pi / 3.0     # 60 degrees
        # Randomly choose left or right region
        if np.random.rand() < 0.5:
            # Left region
            theta = np.random.uniform(np.pi - angle_range_outer, np.pi - angle_range_inner)
        else:
            # Right region
            theta = np.random.uniform(np.pi + angle_range_inner, np.pi + angle_range_outer)

    else: # Stage 3+: Full circle
        # Stage 3: ±180°, full circumference, learn U-turns
        theta = np.random.uniform(-np.pi, np.pi)
    
    # Calculate coordinates
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    
    # Round coordinates to one decimal place
    return np.round(np.array([x, y]), 1)

def get_or_create_reward_pos(env_id):
    """Get or create reward position for environment"""
    if env_id not in env_reward_positions:
        new_pos = generate_new_reward_pos()
        env_reward_positions[env_id] = new_pos
        log_message(f"Generated initial reward position for new environment {env_id} (Stage {TRAINING_STAGE}): {np.round(new_pos, 3).tolist()}")
    return env_reward_positions[env_id]

def reset_reward_pos(env_id):
    """Generate new reward position for a new episode"""
    new_pos = generate_new_reward_pos()
    env_reward_positions[env_id] = new_pos
    log_message(f"Generated reward position for environment {env_id} new episode (Stage {TRAINING_STAGE}): {np.round(new_pos, 3).tolist()}")
    return new_pos

class MultiEnvStateRecorder:
    def __init__(self, max_history=300, initial_sign_positive=True):
        self.env_lock = threading.Lock()
        self.env_states = {}  # {env_id: deque of state records}
        self.env_actions = {}  # {env_id: action history}
        self.env_ids = {}  # (host, port) -> env_id
        self.next_env_id = 0
        self.max_history = max_history
        # Track action signs for each environment
        self.initial_sign = 1 if initial_sign_positive else -1
        self.action_signs = {}  # Store next action sign for each env {env_id: sign}
        self.max_history = max_history  # Maximum state records to save
        
    def get_env_id(self, request, client_address=None):
        """Get environment ID from request, fallback to address"""
        # Priority: use env_id from request
        if "env_id" in request:
            return request["env_id"]
        
        # Fallback to old method
        if client_address:
            port = client_address[1]
            ip = client_address[0].replace('.', '_')
            client_key = (ip, port)
            if client_key not in self.env_ids:
                env_id = f"env_{self.next_env_id}"
                self.env_ids[client_key] = env_id
                self.next_env_id += 1
                # Initialize sign for new environment
                self.action_signs[env_id] = self.initial_sign
                log_message(f"Identified new environment: {env_id} from {client_address}")
            return self.env_ids[client_key]
        
        return "unknown_env"
    
    def record_state(self, env_id, fish_data, velocities, power_data, rotational_momentums, current_time):
        """Record environment state"""
        with self.env_lock:
            if env_id not in self.env_states:
                self.env_states[env_id] = deque(maxlen=self.max_history)
                self.env_actions[env_id] = deque(maxlen=self.max_history)
            
            # Build state record
            state_record = {
                'time': current_time,
                'timestamp': time.time(),
                'fish_data': fish_data,
                'velocities': velocities,
                'power_data': power_data,
                'rotational_momentums': rotational_momentums, # Record rotational inertia
                'record_count': len(self.env_states[env_id])
            }
            
            self.env_states[env_id].append(state_record)
    
    def get_env_history(self, env_id=None, last_n=10):
        """Get environment history records"""
        with self.env_lock:
            if env_id:
                # Return history for specified environment
                if env_id in self.env_states:
                    states = list(self.env_states[env_id])[-last_n:]
                    actions = list(self.env_actions[env_id])[-last_n:]
                    return {
                        'env_id': env_id,
                        'state_count': len(self.env_states[env_id]),
                        'action_count': len(self.env_actions[env_id]),
                        'recent_states': states,
                        'recent_actions': actions
                    }
                else:
                    return {'error': f'Environment {env_id} does not exist'}
            else:
                # Return summary for all environments
                summary = {}
                for eid in self.env_states:
                    summary[eid] = {
                        'state_count': len(self.env_states[eid]),
                        'action_count': len(self.env_actions[eid]),
                        'latest_time': self.env_states[eid][-1]['time'] if self.env_states[eid] else None,
                        'latest_action_time': self.env_actions[eid][-1]['time'] if self.env_actions[eid] else None
                    }
                return summary
    
    def save_env_states_separately(self, output_dir="env_data"):
        """Save state files separately for each environment"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_files = []
        current_time = int(time.time())
        
        with self.env_lock:
            for env_id in self.env_states:
                if len(self.env_states[env_id]) == 0:
                    continue
                    
                # State file
                state_filename = f"{output_dir}/states_{env_id}_{current_time}.json"
                state_data = {
                    'env_id': env_id,
                    'export_time': time.time(),
                    'total_records': len(self.env_states[env_id]),
                    'states': list(self.env_states[env_id])
                }
                
                with open(state_filename, 'w', encoding='utf-8') as f:
                    json.dump(state_data, f, indent=2, ensure_ascii=False)
                
                saved_files.append(state_filename)
                log_message(f"Environment {env_id} states saved to {state_filename} ({len(self.env_states[env_id])} records)")
        
        return saved_files
    
    def save_env_actions_separately(self, output_dir="env_data"):
        """Save action history files separately for each environment"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_files = []
        current_time = int(time.time())
        
        with self.env_lock:
            for env_id in self.env_actions:
                if len(self.env_actions[env_id]) == 0:
                    continue
                    
                # Action file
                action_filename = f"{output_dir}/actions_{env_id}_{current_time}.json"
                action_data = {
                    'env_id': env_id,
                    'export_time': time.time(),
                    'total_actions': len(self.env_actions[env_id]),
                    'actions': list(self.env_actions[env_id])
                }
                
                with open(action_filename, 'w', encoding='utf-8') as f:
                    json.dump(action_data, f, indent=2, ensure_ascii=False)
                
                saved_files.append(action_filename)
                log_message(f"Environment {env_id} action history saved to {action_filename} ({len(self.env_actions[env_id])} records)")
        
        return saved_files
    
    def save_all_env_data(self, output_dir="env_data"):
        """Save all environment state and action data"""
        state_files = self.save_env_states_separately(output_dir)
        action_files = self.save_env_actions_separately(output_dir)
        
        # Create summary information
        summary_filename = f"{output_dir}/summary_{int(time.time())}.json"
        summary_data = {
            'export_time': time.time(),
            'total_environments': len(self.env_states),
            'state_files': state_files,
            'action_files': action_files,
            'environment_summary': {}
        }
        
        with self.env_lock:
            for env_id in self.env_states:
                summary_data['environment_summary'][env_id] = {
                    'state_count': len(self.env_states[env_id]),
                    'action_count': len(self.env_actions.get(env_id, [])),
                    'latest_state_time': self.env_states[env_id][-1]['time'] if self.env_states[env_id] else None,
                    'latest_action_time': self.env_actions[env_id][-1]['time'] if env_id in self.env_actions and self.env_actions[env_id] else None
                }
        
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        log_message(f"Summary information saved to {summary_filename}")
        log_message(f"Saved {len(state_files)} state files, {len(action_files)} action files")
        
        return {
            'summary_file': summary_filename,
            'state_files': state_files,
            'action_files': action_files
        }

    def record_action(self, env_id, action_record):
        """
        Record action to history with automatic sign alternation.
        Only historical action signs change, returned action values remain unchanged.
        """
        # Get current environment sign
        current_sign = self.action_signs.get(env_id, self.initial_sign)
        
        # Create copy to apply sign without affecting original record
        signed_action_record = action_record.copy()
        signed_action_record['amplitude'] *= current_sign
        
        # Add signed record to history
        if env_id not in self.env_actions:
            self.env_actions[env_id] = deque(maxlen=self.max_history)
        self.env_actions[env_id].append(signed_action_record)
        
        # Flip sign for next action
        self.action_signs[env_id] = -current_sign

    def clear_env_history(self, env_id):
        """Clear history data for specified environment and reset sign"""
        if env_id in self.env_states:
            self.env_states[env_id].clear()
        if env_id in self.env_actions:
            self.env_actions[env_id].clear()
        # Reset sign to ensure consistent starting sign for each new episode
        self.action_signs[env_id] = self.initial_sign
        log_message(f"Cleared environment {env_id} history data and action signs")

# Global recorder, first action from bottom to top recorded as True, otherwise False
state_recorder = MultiEnvStateRecorder(max_history=HISTORY_LENGTH, initial_sign_positive=True)

state_processor = StateProcessor(
    pressure_dim=PRESSURE_DIM, 
    action_history_length=ACTION_HISTORY_LENGTH,
    history_length=HISTORY_LENGTH,
    reward_scale_factor=2, # Reward scaling factor
    max_distance=4.0 # Set reasonable value based on environment
)

# PPG Agent and training coordination (Note: PPG has degenerated to PPO due to aux_loss=0)
# Initialize PPG agent
ppg_agent = PPGAgent(
    action_dim=ACTION_DIM, 
    gru_hidden_dims=FEATURE_DIMS, 
    pressure_input_dim=PRESSURE_DIM, 
    action_history_length=ACTION_HISTORY_LENGTH, 
    lr_policy=0.001,
    lr_value=0.0015,
    gamma=0.999,
    gae_lambda=0.95,
    e_policy=4,
    e_value=10,
    e_aux=0,  # aux_loss=0 means PPG degenerates to PPO
    value_loss_coef=0.1, # Balance auxiliary value loss gradients
    minibatch_size=32
)

# Transition buffers: store (s_t, a_t, log_prob_t, v_t) waiting for reward r_t
# Format: {env_id: {'state': s_t, 'action': a_t, ...}}
transition_buffers = defaultdict(deque)

# Store latest calculated reward value for each environment
latest_rewards = defaultdict(float)

# Track terminated environments
terminated_envs = set()

# Training coordination parameters
# Collect trajectories by environment ID
trajectory_memory = defaultdict(list)

def log_message(message):
    """Log messages"""
    with open("model_server_log.txt", "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

# Episode counters
episode_counters = defaultdict(int)  # {env_id: episode_count}

# Store total rewards for each environment's episodes
episode_total_rewards = defaultdict(list)

# Training counter
training_counter = 0

def is_close_to_d_obstacle(fish_points, obstacle_pos, obstacle_radius=0.5, proximity_threshold=0.1):
    """
    Check if any point on fish body is close to D-shaped cylindrical obstacle.
    Obstacle is a semi-circle stretched on y-axis with flat side in positive x direction.
    """
    cx, cy = obstacle_pos
    total_radius = obstacle_radius + proximity_threshold
    min_dist_sq = float('inf')

    for point in fish_points:
        px, py = point
        dist_sq = (px - cx)**2 + (py - cy)**2

        # Check if point is in semi-circle region
        if px <= cx:
            # Point is left of obstacle (semi-circle side)
            if dist_sq < total_radius**2:
                # Collision!
                return True, np.sqrt(dist_sq) - obstacle_radius
        else:
            # Point is right of obstacle (rectangle side)
            # Check if point is in rectangular danger zone
            if abs(py - cy) < total_radius and px < (cx + proximity_threshold):
                 # Collision!
                return True, px - cx

        min_dist_sq = min(min_dist_sq, dist_sq)

    # If no collision, return False and distance to obstacle center
    return False, np.sqrt(min_dist_sq) - obstacle_radius
    
def check_episode_termination(state_history, reward_pos):
    """Check if episode has ended"""
    done = False
    success = False
    reason = ""
    
    if not state_history:
        return done, success, reason
        
    try:
        latest_state = state_history[-1]
        fish_data = latest_state.get('fish_data', [])
        
        if fish_data and len(fish_data[0].get('coordinates', [])) >= 5:
            fish_origin_coords = fish_data[0]['coordinates'][4]
            x, y = fish_origin_coords[0], fish_origin_coords[1]
            fish_pos = np.array([x, y])

            # Check success condition
            dist_to_reward = np.linalg.norm(fish_pos - reward_pos)
            if dist_to_reward < 0.4:
                done = True
                success = True
                reason = f"Reached reward area (distance={dist_to_reward:.2f})"
                return done, success, reason

            # Out of bounds: too far from reward target
            if dist_to_reward > 4.0:
                done = True
                success = False
                reason = f"Too far from reward point({dist_to_reward:.2f} > 4.0), no chance to approach"
                return done, success, reason
            
            # Boundary check: check distance from center (-2.5, 0) is less than 4.0
            center_point = np.array([-4.0, 0.0])
            distance_from_center = np.linalg.norm(fish_pos - center_point)
            if distance_from_center > 4.0:
                done = True
                reason = f"Out of bounds (distance from center {distance_from_center:.2f} > 4.0)"
                return done, success, reason
                
    except (IndexError, TypeError, KeyError):
        pass
        
    return done, success, reason

def check_and_trigger_training():
    """Check and trigger training"""
    global training_counter

    if not trajectory_memory:
        return

    # Calculate total trajectory length across all environments
    total_trajectory_length = sum(len(traj) for traj in trajectory_memory.values())

    # Trigger training when total trajectory length reaches threshold
    if total_trajectory_length < TOTAL_TRAJECTORY_LENGTH_TO_UPDATE:
        return

    log_message(f"--- Triggering update! Total trajectory length({total_trajectory_length}) >= threshold({TOTAL_TRAJECTORY_LENGTH_TO_UPDATE}) ---")

    # Collect all trajectories (including completed episode trajectories)
    training_batch = []
    for env, traj in trajectory_memory.items():
        training_batch.extend(traj)
        log_message(f"    - Pulled environment {env} trajectory, length: {len(traj)}")
    
    # Clear all trajectories
    trajectory_memory.clear()
    log_message("--- All trajectory caches cleared ---")
    
    # Execute training - directly pass batch data
    log_message(f"--- Starting PPG update, total steps: {len(training_batch)} ---")
    # PPG update method returns four values: three losses and policy details dict
    policy_loss, value_loss, aux_loss, policy_details = ppg_agent.update(training_batch)
    
    if policy_loss is not None:
        training_counter += 1
        
        # Log policy update details for each epoch
        if policy_details:
            log_message("--- Policy Update Details ---")
            
            # Print invariant information (from first epoch)
            first_epoch_details = policy_details[0]
            if "advantages_mean" in first_epoch_details:
                # Print non-normalized advantage statistics
                if "unnormalized_adv_mean" in first_epoch_details:
                    log_message(f"  Advantage (raw) | mean: {first_epoch_details['unnormalized_adv_mean']:.4f}, std: {first_epoch_details['unnormalized_adv_std']:.4f}, "
                                f"min: {first_epoch_details['unnormalized_adv_min']:.4f}, max: {first_epoch_details['unnormalized_adv_max']:.4f}")
                
                log_message(f"  Advantage (norm)| mean: {first_epoch_details['advantages_mean']:.4f}, std: {first_epoch_details['advantages_std']:.4f}")
                log_message(f"  Batch Info      | Old Log Probs (mean): {first_epoch_details['old_log_probs_mean']:.4f}")

            log_message("  --- Epoch by Epoch Update ---")
            # Log changes for each epoch
            for details in policy_details:
                log_message(f"  Epoch {details['epoch']}: Loss={details['policy_loss']:.6f}, Ratio(mean)={details['ratio_mean']:.4f}, NewLogProbs(mean)={details['new_log_probs_mean']:.4f}")
            
            log_message("-----------------------------")

        # Log auxiliary loss in PPG
        log_message(f"--- PPG update completed (Round {training_counter}) --- Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Aux Loss: {aux_loss:.4f}")
        # Pass all three loss histories to plotting function
        save_loss_plot(ppg_agent.policy_loss_history, ppg_agent.value_loss_history, ppg_agent.aux_loss_history)
        
        # Save model every 10 training sessions
        if training_counter % 10 == 0:
            save_checkpoint(training_counter)

def check_and_advance_stage():
    """Check and advance training stage"""
    global TRAINING_STAGE
    
    # Maximum stage is 3, no further advancement
    if TRAINING_STAGE >= 3:
        return
        
    # Need at least one environment's data
    if not stage_eval_history:
        return
    
    # Only evaluate when environment history is full for more stable assessment
    envs_with_full_history = []
    for env_id, history in stage_eval_history.items():
        if len(history) == STAGE_EVALUATION_WINDOW:
            envs_with_full_history.append(history)

    # Need at least one environment with full history for evaluation
    if not envs_with_full_history:
        return

    total_successes = 0
    total_episodes = 0
    for history in envs_with_full_history:
        total_successes += sum(history)  # True counts as 1, False as 0
        total_episodes += len(history)
        
    if total_episodes == 0:
        return
        
    overall_success_rate = total_successes / total_episodes
    
    if overall_success_rate >= STAGE_UPGRADE_THRESHOLD:
        old_stage = TRAINING_STAGE
        TRAINING_STAGE += 1
        log_message(f"��🚀🚀 Training stage upgraded! Stage {old_stage} -> {TRAINING_STAGE} 🚀🚀🚀")
        log_message(f"   - Reason: Overall success rate {overall_success_rate:.2f} >= threshold {STAGE_UPGRADE_THRESHOLD}")
        
        # Clear all environment evaluation history for new stage assessment
        stage_eval_history.clear()
        log_message("   - Stage evaluation history cleared for new stage assessment.")

def save_checkpoint(training_step):
    """Save model checkpoint"""
    try:
        # Create checkpoints directory
        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Build checkpoint filename
        checkpoint_path = f"{checkpoint_dir}/model_checkpoint_step_{training_step}.pth"
        
        # Save complete model state
        checkpoint = {
            'model_state_dict': ppg_agent.policy.state_dict(),
            # Save both optimizers' states
            'policy_optimizer_state_dict': ppg_agent.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': ppg_agent.value_optimizer.state_dict(),
            
            # Training configuration
            'training_step': training_step,
            'action_dim': ACTION_DIM,
            'gru_hidden_dims': FEATURE_DIMS,
            'pressure_dim': PRESSURE_DIM,
            'action_history_length': ACTION_HISTORY_LENGTH,
            'history_length': HISTORY_LENGTH,
            
            # Training parameters
            'gamma': ppg_agent.gamma,
            'eps_clip': ppg_agent.eps_clip,
            'gae_lambda': ppg_agent.gae_lambda,
            'value_loss_coef': ppg_agent.value_loss_coef,
            'beta_clone': ppg_agent.beta_clone,
            
            # Save three loss histories
            'policy_loss_history': ppg_agent.policy_loss_history,
            'value_loss_history': ppg_agent.value_loss_history,
            'aux_loss_history': ppg_agent.aux_loss_history,
            
            # Environment state
            'episode_counters': dict(episode_counters),
            'min_trajectory_length': TOTAL_TRAJECTORY_LENGTH_TO_UPDATE,
            
            # Timestamp
            'save_time': time.time(),
            'save_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        log_message(f"✅ Model checkpoint saved: {checkpoint_path}")
        log_message(f"   - Training steps: {training_step}")
        log_message(f"   - Policy Loss: {ppg_agent.policy_loss_history[-1]:.4f}")
        log_message(f"   - Value Loss: {ppg_agent.value_loss_history[-1]:.4f}")
        if ppg_agent.aux_loss_history:
             log_message(f"   - Aux Loss: {ppg_agent.aux_loss_history[-1]:.4f}")
        
    except Exception as e:
        log_message(f"❌ Failed to save checkpoint: {str(e)}")

def load_checkpoint(checkpoint_path):
    """Load model checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path)
        
        # Restore model parameters
        ppg_agent.policy.load_state_dict(checkpoint['model_state_dict'])
        ppg_agent.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        # Compatible with old single optimizer format
        if 'value_optimizer_state_dict' in checkpoint:
            ppg_agent.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        else:
            log_message("⚠️ value_optimizer_state_dict not found, value network optimizer state not loaded.")

        
        # Restore training history
        ppg_agent.policy_loss_history = checkpoint['policy_loss_history']
        ppg_agent.value_loss_history = checkpoint['value_loss_history']
        # Compatible with old checkpoints
        if 'aux_loss_history' in checkpoint:
            ppg_agent.aux_loss_history = checkpoint['aux_loss_history']
        
        # Load hyperparameters if they exist
        if 'value_loss_coef' in checkpoint:
            ppg_agent.value_loss_coef = checkpoint['value_loss_coef']
        if 'beta_clone' in checkpoint:
            ppg_agent.beta_clone = checkpoint['beta_clone']
        
        # Restore global counters
        global training_counter
        training_counter = checkpoint['training_step']
        
        # Restore episode counters
        global episode_counters
        episode_counters.update(checkpoint['episode_counters'])
        
        log_message(f"✅ Model checkpoint loaded successfully: {checkpoint_path}")
        log_message(f"   - Training steps: {training_counter}")
        log_message(f"   - Save time: {checkpoint['save_timestamp']}")
        
        return True
        
    except Exception as e:
        log_message(f"❌ Failed to load checkpoint: {str(e)}")
        return False

def handle_state_record(request, client_address):
    """Only responsible for recording environment state flow, not handling learning logic"""
    try:
        env_id = state_recorder.get_env_id(request, client_address)
        
        # If environment has ended, clean up history data
        if env_id in terminated_envs:
            log_message(f"Detected environment {env_id} restart, cleaning history data...")
            
            # Call clear_env_history to atomically clear history and reset action signs
            state_recorder.clear_env_history(env_id)

            # Reset previous episode's leftover reward value to ensure new episode starts from 0
            if env_id in latest_rewards:
                del latest_rewards[env_id]

            terminated_envs.discard(env_id)
            log_message(f"Environment {env_id} leftover trajectories and reward values cleared for new episode")

        # Only record state
        fish_data = request.get("fish_data", [])
        velocities = request.get("velocities", [])
        power_data = request.get("power_data", [])
        rotational_momentums = request.get("angular_velocities", []) # Extract rotational inertia
        current_time = request.get("time", 0.0)
        
        state_recorder.record_state(env_id, fish_data, velocities, power_data, rotational_momentums, current_time)
        
        # Record detailed analysis data
        # Ensure correct reward position before recording
        reward_pos = get_or_create_reward_pos(env_id)
        record_episode_data(env_id, current_time, fish_data, velocities, power_data, rotational_momentums, reward_pos)
        
        return {"status": "recorded", "env_id": env_id, "time": current_time}
    except Exception as e:
        log_message(f"State recording failed: {str(e)}")
        return {"status": "error", "error": str(e)}

def record_episode_data(env_id, current_time, fish_data, velocities, power_data, rotational_momentums, reward_pos):
    """Record detailed episode data to separate log files"""
    try:
        # Create env_log directory
        log_dir = "env_log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            log_message(f"✅ Created log directory: {log_dir}")
        
        # Prepare log filename with episode count
        current_episode = episode_counters[env_id]
        if current_episode == 0:  # First run
            episode_counters[env_id] = 1
            current_episode = 1
        
        log_filename = f"{log_dir}/{env_id}_episode_{current_episode}.log"
        
        # Check if file exists, create and write header if not
        if not os.path.exists(log_filename):
            with open(log_filename, 'w') as f:
                f.write("time,fish_x,fish_y,vel_x,vel_y,dist_to_reward,angle_to_reward,vel_projection,constraint_power,inertia_power,total_power,reward_metric,reward,cos_angle_to_reward,sin_angle_to_reward,rotational_momentum_z\n")

        
        # Check data completeness
        if not fish_data or not velocities or not power_data:
            log_message(f"❌ Incomplete data, skipping record")
            return
        
        if len(fish_data) == 0:
            log_message(f"❌ fish_data is empty list, skipping record")
            return
            
        if len(fish_data[0].get('coordinates', [])) < 5:
            coords_len = len(fish_data[0].get('coordinates', []))
            log_message(f"❌ Insufficient coordinate points, need 5 but only have {coords_len}, skipping record")
            return
        
        # Fish body center point (index 4)
        fish_origin_coords = fish_data[0]['coordinates'][4] # Use index 4 as origin, consistent with state_processor and termination check
        fish_x, fish_y = fish_origin_coords[0], fish_origin_coords[1]
        fish_pos = np.array([fish_x, fish_y])
        
        # Instantaneous velocity components
        vel_x = velocities[0]['x']
        vel_y = velocities[0]['y']
        velocity_vec = np.array([vel_x, vel_y])
        
        # Distance and angle to reward point
        vec_to_reward = reward_pos - fish_pos
        dist_to_reward = np.linalg.norm(vec_to_reward)
        
        # Calculate angle (world coordinate system)
        angle_to_reward = np.degrees(np.arctan2(vec_to_reward[1], vec_to_reward[0]))
        
        # Velocity projection in reward direction
        if dist_to_reward > 1e-6:
            vel_projection = np.dot(velocity_vec, vec_to_reward) / dist_to_reward
        else:
            vel_projection = 0.0
        
        # Power data
        constraint_power = np.linalg.norm([power_data[0]['constraint']['x'], power_data[0]['constraint']['y']])
        inertia_power = np.linalg.norm([power_data[0]['inertia']['x'], power_data[0]['inertia']['y']])
        total_power = constraint_power + inertia_power
        
        # Rotational inertia
        rot_mom_z = 0.0
        if rotational_momentums:
            try:
                rot_mom_z = rotational_momentums[0]['z'] * 1000.0
            except (IndexError, KeyError):
                rot_mom_z = 0.0

        # Use reward calculation method consistent with training logic
        # Introduce distance-inverse scaling
        distance_scaling_factor = 1.0 / (dist_to_reward + 1.0)
        
        # New reward metric = (effective velocity * distance scaling) - energy cost
        reward_metric = float(vel_projection * distance_scaling_factor - total_power)
        
        # Calculate fish head orientation vs reward direction cos and sin values
        # Use orientation calculation method consistent with state_processor
        # Calculate fish body local coordinate system base axes
        fish_points = np.array([item for item in fish_data[0]['coordinates']])
        p1 = fish_points[0]
        p5 = fish_points[4]
        
        x_axis_vec = p1 - p5
        norm_x = np.linalg.norm(x_axis_vec)
        if norm_x < 1e-6:
            x_axis = np.array([1, 0])
        else:
            x_axis = x_axis_vec / norm_x

        # Fish head orientation vector defined as local coordinate system y-axis (perpendicular to x-axis)
        heading_vec_normalized = np.array([-x_axis[1], x_axis[0]])

        # Define vector to reward point (vec_to_reward already calculated above)
        if dist_to_reward > 1e-6:
            vec_to_reward_normalized = vec_to_reward / dist_to_reward
        else:
            vec_to_reward_normalized = np.array([0.0, 0.0])

        # Calculate cos and sin
        cos_angle_to_reward = np.dot(heading_vec_normalized, vec_to_reward_normalized)
        sin_angle_to_reward = np.cross(heading_vec_normalized, vec_to_reward_normalized)

        # Get reward value associated with previous action
        reward_value = latest_rewards[env_id]

        # Build data row
        data_line = f"{current_time:.6f},{fish_x:.6f},{fish_y:.6f},{vel_x:.6f},{vel_y:.6f},{dist_to_reward:.6f},{angle_to_reward:.6f},{vel_projection:.6f},{constraint_power:.6f},{inertia_power:.6f},{total_power:.6f},{reward_metric:.6f},{reward_value:.6f},{cos_angle_to_reward:.6f},{sin_angle_to_reward:.6f},{rot_mom_z:.6f}\n"
        
        # Write data
        with open(log_filename, 'a') as f:
            f.write(data_line)
            
    except Exception as e:
        log_message(f"❌ Failed to record episode data {env_id}: {str(e)}")
        import traceback
        log_message(f"❌ Detailed error info: {traceback.format_exc()}")

def handle_control_request(request, client_address):
    """Handle control requests from C++, return actions"""
    try:
        # Record latest received state
        env_id = state_recorder.get_env_id(request, client_address)
        fish_data = request.get("fish_data", [])
        velocities = request.get("velocities", [])
        power_data = request.get("power_data", [])
        
        # Compatible with old and new key names
        rotational_momentums = request.get("angular_velocities", [])
        
        current_time = request.get("time", 0.0)
        
        # Get or create current environment's reward position
        reward_pos = get_or_create_reward_pos(env_id)
        
        # Dynamically update state_processor's reward position so all internal calls use latest position
        state_processor.reward_pos = reward_pos
        
        # Return safe action if environment has ended
        if env_id in terminated_envs:
            log_message(f"Refusing to provide new action for terminated environment {env_id}")
            return {"amplitudes": [1.0], "T_n_values": [1.0]}

        # Check if historical data is sufficient
        state_history = state_recorder.env_states.get(env_id, deque())
        action_history = state_recorder.env_actions.get(env_id, deque())
        
        if not state_processor.is_data_sufficient(state_history, action_history):
            log_message(f"Environment {env_id} insufficient historical data, returning fixed action (states:{len(state_history)}/{HISTORY_LENGTH}, actions:{len(action_history)}/{ACTION_HISTORY_LENGTH})")
            
            # Record fixed action to history
            action_record = {
                'time': current_time,
                'amplitude': 0.4,
                'T_n': 1.0,
                'action_count': len(state_recorder.env_actions.get(env_id, []))
            }
            state_recorder.record_action(env_id, action_record)
            
            return {"amplitudes": [0.4], "T_n_values": [1.0]}

        # === Part 1: Process t-2 moment transition ===
        # Need at least 2 action history and 2 buffer items to calculate and assign reward for a_{t-2}
        if len(action_history) >= 2 and len(transition_buffers[env_id]) >= 2:
            # 1. Calculate reward for a_{t-2}. Window is [t(a_{t-2}), t(s_t)]
            reward = state_processor.calculate_reward(state_history, action_history, reward_pos)
            latest_rewards[env_id] = reward

            # 2. Pop a_{t-2} transition data
            completed_transition_data = transition_buffers[env_id].popleft()
            state = completed_transition_data['state']
            action = completed_transition_data['action_indices']
            log_prob = completed_transition_data['log_probs']
            value = completed_transition_data['value']

            # 3. Get next_state (s_{t-1}) and done flag
            next_step_data = transition_buffers[env_id][0]
            next_state = next_step_data['state']
            done = next_step_data['done']

            # 4. Apply terminal reward if s_{t-1} is endpoint
            if done:
                success = next_step_data['success']
                reward = 10.0 if success else -1.0
                log_message(f"Applied terminal reward {reward:.1f} to transition ending in {env_id}")

            # 5. Build and store complete transition record
            transition = (state, action, log_prob, reward, done, value, next_state)
            trajectory_memory[env_id].append(transition)

            # 6. If episode ends, execute all cleanup and logging work
            if done:
                reason = next_step_data['reason']
                success = next_step_data['success']
                
                # a_{t-1} data still in buffer but discarded due to episode end
                transition_buffers[env_id].clear()

                finished_episode_number = episode_counters[env_id]
                log_message(f"   Environment {env_id} episode {finished_episode_number} ended: {reason}")
                
                # Use correct numbering for plotting
                try:
                    log_filename = f"env_log/{env_id}_episode_{finished_episode_number}.log"
                    plot_filename = f"env_log/{env_id}_episode_{finished_episode_number}_analysis.png"
                    
                    # Get additional plotting info from state_processor
                    obstacle_pos = state_processor.obstacle_pos

                    # Call plotting function
                    plot_episode_data(log_filename, plot_filename, reward_pos, None)

                    # Verify image generation for more reliable logging
                    if os.path.exists(plot_filename):
                        log_message(f"�� Generated analysis plot for episode {finished_episode_number}: {plot_filename}")
                        # Calculate and record episode total reward
                        total_reward = calculate_total_reward_from_log(log_filename)
                        episode_total_rewards[env_id].append(total_reward)
                        
                        # Call new plotting function for current environment alone
                        rewards_plot_filename = plot_env_cumulative_rewards(env_id, episode_total_rewards[env_id])
                        if rewards_plot_filename:
                            log_message(f"�� Episode {finished_episode_number} total reward: {total_reward:.2f}. Updated cumulative reward plot: {rewards_plot_filename}")
                    else:
                        log_message(f"⚠️ Failed to generate episode {finished_episode_number} analysis plot. Check if log file '{log_filename}' exists and is non-empty.")

                except Exception as e:
                    log_message(f"❌ Error generating episode analysis plot: {e}")

                # Increment counter for next episode
                episode_counters[env_id] += 1
                
                # Record episode result and check if can upgrade
                stage_eval_history[env_id].append(success)
                check_and_advance_stage()
                terminated_envs.add(env_id)

                if success:
                    log_message(f"--- Episode successful: {env_id}")
                    reset_reward_pos(env_id)
                else:
                    log_message(f"--- Episode failed: {env_id}")
                
                # Training and return safe action
                check_and_trigger_training()
                return {"amplitudes": [0.0], "T_n_values": [1.0]}
            # Check if can trigger training if episode not ended (based on steps)
            check_and_trigger_training()
        # === Part 2: Generate current action (a_t) ===
        current_state = state_processor.process(state_history, action_history, reward_pos)

        # Unpack state for recording
        pressure_history, latest_visual_info, final_feature_vector = current_state

        # PPO selects action a_t
        action_index, log_prob, value, logits, probs = ppg_agent.select_action(current_state)

        # Check if current state is terminal state
        # state_history now contains latest state data
        done, success, reason = check_episode_termination(state_history, reward_pos)

        # Store current step (t) information in buffer
        transition_buffers[env_id].append({
            'state': current_state,
            'action_indices': action_index,
            'log_probs': log_prob,
            'value': value,
            'done': done,
            'success': success,
            'reason': reason
        })

        # Map actions and record to history
        action_values = ACTION_MAP[action_index]
        amplitude, T_n = action_values[0], action_values[1]
        action_record = {
            'time': current_time,
            'amplitude': amplitude,
            'T_n': T_n,
            'action_count': len(action_history)
        }
        state_recorder.record_action(env_id, action_record)
        # Record state details used for decision making
 
        log_message(f"Generated action (Stage {TRAINING_STAGE}): {env_id}, time={current_time:.3f}, amplitude={amplitude}, T_n={T_n}")
        log_message(f"  - Action Index: {action_index}")
        log_message(f"  - Logits: {[f'{x:.3f}' for x in logits]}")
        log_message(f"  - Probs: {[f'{p:.3f}' for p in probs]}")
        if hasattr(latest_visual_info, 'flatten'):
            log_message(f"  - Visual Info: {[f'{x:.4f}' for x in latest_visual_info.flatten()]}")
        else:
            log_message(f"  - Visual Info: {latest_visual_info}")
        if hasattr(final_feature_vector, 'flatten'):
            log_message(f"  - Feature Vec: {[f'{x:.4f}' for x in final_feature_vector.flatten()]}")
        else:
            log_message(f"  - Feature Vec: {final_feature_vector}")
        
        return {
            "amplitudes": [amplitude],
            "T_n_values": [T_n],
            "env_id": env_id
        }
        
    except Exception as e:
        log_message(f"Action generation failed: {str(e)}")
        return {"amplitudes": [1.0], "T_n_values": [1.0]}  # Fixed action

def handle_query_request(request):
    """Handle query requests"""
    try:
        query_type = request.get("query", "summary")
        env_id = request.get("env_id", None)
        last_n = request.get("last_n", 10)
        output_dir = request.get("output_dir", "env_data")
        
        if query_type == "history":
            return state_recorder.get_env_history(env_id, last_n)
        elif query_type == "summary":
            return state_recorder.get_env_history()
        elif query_type == "save_all":
            # Save all environment data
            result = state_recorder.save_all_env_data(output_dir)
            return {"status": "saved", "result": result}
        elif query_type == "save_states":
            # Save only state files
            files = state_recorder.save_env_states_separately(output_dir)
            return {"status": "saved", "state_files": files}
        elif query_type == "save_actions":
            # Save only action files
            files = state_recorder.save_env_actions_separately(output_dir)
            return {"status": "saved", "action_files": files}
        else:
            return {"error": "Unknown query type"}
            
    except Exception as e:
        return {"error": str(e)}

# Global variables
is_handling_request = False
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    log_message("Received exit signal, preparing to shutdown server")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def start_server(port=9999):
    global shutdown_requested
    
    log_message(f"Starting state recording server, port: {port}")
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen(10)
    
    log_message(f"Server listening on 0.0.0.0:{port}")

    
    try:
        while not shutdown_requested:
            try:
                client_socket, address = server_socket.accept()
                client_socket.settimeout(3.0)
                
                # Receive data
                data = b""
                while True:
                    try:
                        chunk = client_socket.recv(4096)
                        if not chunk or data.endswith(b"\n"):
                            break
                        data += chunk
                    except socket.timeout:
                        break
                
                if data:
                    try:
                        request = json.loads(data.decode('utf-8').strip())
                        request_type = request.get("request_type")
                        
                        if request_type == "state_record":
                            response = handle_state_record(request, address)
                        elif request_type == "query":
                            response = handle_query_request(request)
                        elif request_type == "state_record_batch":
                            # Get correct environment ID from batch request
                            batch_env_id = state_recorder.get_env_id(request, address)
                            records = request.get("records", [])
                            for record in records:
                                # Inject correct env_id into each individual record
                                record['env_id'] = batch_env_id
                                handle_state_record(record, address)
                            response = {"status": "batch_recorded", "count": len(records)}
                        else:  # Default to control request
                            response = handle_control_request(request, address)
                        
                        # Send response
                        response_json = json.dumps(response) + "\n"
                        client_socket.sendall(response_json.encode('utf-8'))
                        
                    except json.JSONDecodeError as e:
                        log_message(f"JSON parsing error: {str(e)}")
                        error_response = {"error": "Invalid JSON"}
                        client_socket.sendall((json.dumps(error_response) + "\n").encode('utf-8'))
                
                client_socket.close()
                
            except Exception as e:
                log_message(f"Connection handling error: {str(e)}")
                
    except KeyboardInterrupt:
        pass
    finally:
        server_socket.close()
        log_message("Server has been shut down")


def set_seeds(seed_value):
    """
    Set all related random seeds to ensure experiment reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    # If using GPU in the future, also need these two lines
    # torch.cuda.manual_seed(seed_value)
    # torch.cuda.manual_seed_all(seed_value) # for multi-GPU

if __name__ == "__main__":
    # Set seeds before all operations!
    SEED = 2566  
    set_seeds(SEED)
    log_message(f"✅ All random seeds set to: {SEED}")

    parser = argparse.ArgumentParser(description='State recording server')
    parser.add_argument('--port', type=int, default=9999, help='Port number')
    parser.add_argument('--auto-save', type=int, default=0, help='Auto-save interval (seconds, 0 means no auto-save)')
    parser.add_argument('--load-checkpoint', type=str, default='', help='Load checkpoint file path')
    args = parser.parse_args()
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        if os.path.exists(args.load_checkpoint):
            load_checkpoint(args.load_checkpoint)
        else:
            log_message(f"⚠️ Checkpoint file does not exist: {args.load_checkpoint}")
    
    # Periodic save thread
    if args.auto_save > 0:
        def auto_save_thread():
            while not shutdown_requested:
                time.sleep(args.auto_save)
                if not shutdown_requested:
                    state_recorder.save_all_env_data()
        
        save_thread = threading.Thread(target=auto_save_thread, daemon=True)
        save_thread.start()
    
    try:
        start_server(port=args.port)
    except Exception as e:
        log_message(f"Program exception: {str(e)}")
    finally:
        # Save once when program ends
        state_recorder.save_all_env_data()
        # Save final checkpoint
        save_checkpoint(training_counter)
        sys.exit(1)