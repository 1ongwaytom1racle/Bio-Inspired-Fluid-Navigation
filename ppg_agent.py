import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.distributions.kl as kl
import math

# Action mapping: Convert network output indices to physical values (amplitude, T_n)
ACTION_MAP = [
    (0.2, 0.6), (0.8, 0.6), (1.0, 0.6), (1.4, 0.6),
    (0.2, 0.8), (0.8, 0.8), (1.0, 0.8), (1.4, 0.8),
    (0.2, 1.0), (0.8, 1.0), (1.0, 1.0), (1.4, 1.0),
]
ACTION_DIM = len(ACTION_MAP)

class ActorCritic(nn.Module):
    def __init__(self, action_dim, gru_hidden_dims, pressure_input_dim=15, action_history_length=4):
        super(ActorCritic, self).__init__()

        # Input dimensions
        visual_input_dim = 3
        # Action history (N-1) + v_parallel (1) + v_perp (1) + wz (1)
        action_velocity_dim = (action_history_length - 1) + 3

        # Actor (Policy) network
        actor_feature_dim = visual_input_dim + action_velocity_dim
        self.actor_net = nn.Sequential(
            nn.Linear(actor_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.action_head = nn.Linear(32, action_dim)
        self.aux_value_head = nn.Linear(32, 1)  # PPG auxiliary value head

        # Critic (Value) network
        critic_feature_dim = visual_input_dim + action_velocity_dim
        self.critic_net = nn.Sequential(
            nn.Linear(critic_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward_actor(self, pressure_history, visual_info, action_history):
        # Pressure history is unused in this version
        action_history_flat = action_history.view(action_history.size(0), -1)
        combined_features = torch.cat([visual_info, action_history_flat], dim=1)
        return self.actor_net(combined_features)

    def forward_critic(self, pressure_history, visual_info, action_history):
        # Pressure history is unused in this version
        action_history_flat = action_history.view(action_history.size(0), -1)
        combined_features = torch.cat([visual_info, action_history_flat], dim=1)
        return self.critic_net(combined_features)

    def actor_parameters(self):
        return list(self.actor_net.parameters()) + \
               list(self.action_head.parameters()) + \
               list(self.aux_value_head.parameters())

    def critic_parameters(self):
        return list(self.critic_net.parameters())

    def forward(self, pressure_hist, visual_info, action_hist):
        # pressure_hist is unused in this version
        actor_base = self.forward_actor(pressure_hist, visual_info, action_hist)
        action_logits = self.action_head(actor_base)
        aux_value = self.aux_value_head(actor_base)
        
        main_value = self.forward_critic(pressure_hist, visual_info, action_hist)

        return action_logits, main_value, aux_value


class PPGAgent:
    """
    PPG (Phasic Policy Gradient) agent for action selection and network updates.
    Training is divided into policy and auxiliary phases in the original PPG framework.
    
    Note: This implementation simplifies PPG to function like PPO (Proximal Policy Optimization)
    by setting the auxiliary phase epochs (e_aux) to 0. This removes the auxiliary update phase
    that distinguishes PPG, effectively reducing it to the standard PPO algorithm.
    """
    def __init__(self, action_dim, gru_hidden_dims, pressure_input_dim=40, action_history_length=3, 
                 lr_policy=0.0001, lr_value=0.00015, gamma=0.999, eps_clip=0.2, gae_lambda=0.95,
                 e_policy=4, e_value=10, e_aux=0, beta_clone=1.0, value_loss_coef=0.01, minibatch_size=64):
        
        self.policy = ActorCritic(action_dim, gru_hidden_dims, 
                                pressure_input_dim=pressure_input_dim, 
                                action_history_length=action_history_length)
        
        # Separate optimizers for policy and value networks
        self.policy_optimizer = torch.optim.Adam(self.policy.actor_parameters(), lr=lr_policy)
        self.value_optimizer = torch.optim.Adam(self.policy.critic_parameters(), lr=lr_value)
        
        # RL hyperparameters
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.gae_lambda = gae_lambda
        self.action_history_length = action_history_length
        self.minibatch_size = minibatch_size
        
        # PPG training phases configuration
        self.e_policy = e_policy  # Policy update epochs (fewer)
        self.e_value = e_value    # Value update epochs (more)
        self.e_aux = e_aux        # Auxiliary update epochs
        self.beta_clone = beta_clone  # KL divergence weight in auxiliary phase
        self.value_loss_coef = value_loss_coef  # Auxiliary value loss weight
        
        # Training metrics
        self.policy_loss_history = []
        self.value_loss_history = []
        self.aux_loss_history = []

    def select_action(self, state):
        """Select action based on current state"""
        pressure, visual, actions_hist = state
        pressure = torch.from_numpy(pressure).float().unsqueeze(0)
        visual = torch.from_numpy(visual).float().unsqueeze(0)
        actions = torch.from_numpy(actions_hist).float().unsqueeze(0)

        with torch.no_grad():
            action_logits, state_value, _ = self.policy(pressure, visual, actions)
        
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(logits=action_logits)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), state_value.item(), \
               action_logits.squeeze().cpu().numpy(), action_probs.squeeze().cpu().numpy()

    def update(self, memory_batch):
        if not memory_batch:
            return None, None, None, None

        # Calculate GAE and returns
        all_advantages = []
        all_returns = []

        # Split into independent trajectories
        trajectories = []
        current_traj = []
        for transition in memory_batch:
            current_traj.append(transition)
            if transition[4]:  # If episode done
                trajectories.append(current_traj)
                current_traj = []
        if current_traj:
            trajectories.append(current_traj)

        # Calculate GAE for each trajectory
        for trajectory in trajectories:
            states, actions, old_log_probs, rewards, dones, values, next_states = zip(*trajectory)
            
            advantages = []
            last_gae_lam = 0
            
            values_tensor = torch.tensor(values, dtype=torch.float32)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            dones_tensor = torch.tensor(dones, dtype=torch.float32)

            # Handle terminal state bootstrapping
            if dones[-1]:
                bootstrap_value = 0.0
            else:
                with torch.no_grad():
                    last_next_state = next_states[-1]
                    p, v, a = last_next_state
                    p = torch.from_numpy(p).float().unsqueeze(0)
                    v = torch.from_numpy(v).float().unsqueeze(0)
                    a = torch.from_numpy(a).float().unsqueeze(0)
                    
                    _, bootstrap_value_tensor, _ = self.policy(p, v, a)
                    bootstrap_value = bootstrap_value_tensor.item()

            next_value = bootstrap_value
            for t in reversed(range(len(rewards))):
                next_non_terminal = 1.0 - dones_tensor[t]
                delta = rewards_tensor[t] + self.gamma * next_value * next_non_terminal - values_tensor[t]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                advantages.insert(0, last_gae_lam)
                next_value = values_tensor[t]

            returns = torch.tensor(advantages, dtype=torch.float32) + values_tensor
            
            all_advantages.extend(advantages)
            all_returns.extend(returns)

        # Normalize advantages
        advantages = torch.stack(all_advantages)
        returns = torch.stack(all_returns)
        
        unnormalized_adv_mean = advantages.mean().item()
        unnormalized_adv_std = advantages.std().item()
        unnormalized_adv_max = advantages.max().item()
        unnormalized_adv_min = advantages.min().item()

        if len(memory_batch) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare batch data
        states, actions, old_log_probs, rewards, dones, values, _ = zip(*memory_batch)
        pressure_b = torch.from_numpy(np.array([s[0] for s in states])).float()
        visual_b = torch.from_numpy(np.array([s[1] for s in states])).float()
        action_hist_b = torch.from_numpy(np.array([s[2] for s in states])).float()
        actions_b = torch.tensor(actions, dtype=torch.long)
        old_log_probs_b = torch.tensor(old_log_probs, dtype=torch.float32)
        
        # Save old policy logits for KL divergence calculation
        with torch.no_grad():
            old_action_logits, _, _ = self.policy(pressure_b, visual_b, action_hist_b)
        
        old_dist_detached = Categorical(logits=old_action_logits.detach())


        # PPG training phases
        avg_policy_loss = 0
        avg_value_loss = 0
        avg_aux_loss = 0
        policy_details = []
        
        batch_size = pressure_b.size(0)
        num_minibatches = math.ceil(batch_size / self.minibatch_size)

        # Phase 1: Policy update (PPO loss)
        for i in range(self.e_policy):
            with torch.no_grad():
                action_logits, _, _ = self.policy(pressure_b, visual_b, action_hist_b)
                dist = Categorical(logits=action_logits)
                new_log_probs_full = dist.log_prob(actions_b)
                ratio_full = torch.exp(new_log_probs_full - old_log_probs_b)
                policy_loss_full = -torch.min(ratio_full * advantages, 
                                             torch.clamp(ratio_full, 1 - self.eps_clip, 1 + self.eps_clip) * advantages).mean()

                details = {
                    "epoch": i + 1,
                    "ratio_mean": ratio_full.mean().item(),
                    "ratio_std": ratio_full.std().item(),
                    "new_log_probs_mean": new_log_probs_full.mean().item(),
                    "policy_loss": policy_loss_full.item(),
                }
                if i == 0:
                    details["advantages_mean"] = advantages.mean().item()
                    details["advantages_std"] = advantages.std().item()
                    details["old_log_probs_mean"] = old_log_probs_b.mean().item()
                    details["unnormalized_adv_mean"] = unnormalized_adv_mean
                    details["unnormalized_adv_std"] = unnormalized_adv_std
                    details["unnormalized_adv_max"] = unnormalized_adv_max
                    details["unnormalized_adv_min"] = unnormalized_adv_min
                policy_details.append(details)

            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]

                mb_pressure = pressure_b[mb_indices]
                mb_visual = visual_b[mb_indices]
                mb_action_hist = action_hist_b[mb_indices]
                mb_actions = actions_b[mb_indices]
                mb_old_log_probs = old_log_probs_b[mb_indices]
                mb_advantages = advantages[mb_indices]

                action_logits, _, _ = self.policy(mb_pressure, mb_visual, mb_action_hist)
                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                avg_policy_loss += policy_loss.item()

        # Phase 2: Value update (Main Critic network)
        for _ in range(self.e_value):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]
                
                mb_pressure = pressure_b[mb_indices]
                mb_visual = visual_b[mb_indices]
                mb_action_hist = action_hist_b[mb_indices]
                mb_returns = returns[mb_indices]

                _, new_values, _ = self.policy(mb_pressure, mb_visual, mb_action_hist)
                value_loss = nn.MSELoss()(new_values.squeeze(-1), mb_returns)
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
                avg_value_loss += value_loss.item()
            
        # Phase 3: Auxiliary update (Auxiliary value head + KL divergence)
        for _ in range(self.e_aux):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]

                mb_pressure = pressure_b[mb_indices]
                mb_visual = visual_b[mb_indices]
                mb_action_hist = action_hist_b[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_dist = Categorical(logits=old_action_logits[mb_indices].detach())

                action_logits, _, aux_values = self.policy(mb_pressure, mb_visual, mb_action_hist)
                
                aux_value_loss = nn.MSELoss()(aux_values.squeeze(-1), mb_returns)
                new_dist = Categorical(logits=action_logits)
                kl_loss = kl.kl_divergence(mb_old_dist, new_dist).mean()
                
                aux_loss_total = self.value_loss_coef * aux_value_loss + self.beta_clone * kl_loss

                self.policy_optimizer.zero_grad()
                aux_loss_total.backward()
                self.policy_optimizer.step()
                avg_aux_loss += aux_loss_total.item()

        # Calculate average losses
        total_policy_updates = self.e_policy * num_minibatches
        total_value_updates = self.e_value * num_minibatches
        total_aux_updates = self.e_aux * num_minibatches

        policy_loss_avg = avg_policy_loss / total_policy_updates if total_policy_updates > 0 else 0
        value_loss_avg = avg_value_loss / total_value_updates if total_value_updates > 0 else 0
        aux_loss_avg = avg_aux_loss / total_aux_updates if total_aux_updates > 0 else 0

        self.policy_loss_history.append(policy_loss_avg)
        self.value_loss_history.append(value_loss_avg)
        self.aux_loss_history.append(aux_loss_avg)
        
        return policy_loss_avg, value_loss_avg, aux_loss_avg, policy_details