import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from pettingzoo.mpe import simple_spread_v3
import matplotlib.pyplot as plt
import time
from scipy.stats import wilcoxon
import collections
import multiprocessing as mp

# --- SCRIPT CONFIGURATION ---
USE_SIMPLE_SPREAD = True
SEEDS = [42, 123, 888, 2024, 7, 9999, 3141, 1618, 2718, 404]
NUM_WORKERS = 6
MAX_AGENTS = 5
EPISODES = 20
META_ITERATIONS = 800

# --- Scalability Testing Configuration ---
SCALABILITY_TEST = True  # Set to True to run scalability experiments
SCALABILITY_AGENT_COUNTS = [2, 5, 8, 12, 16]  # Test different agent scales
SCALABILITY_EPISODES = 10  # Reduced episodes for faster testing
SCALABILITY_META_ITERS = 100  # Reduced meta-iterations for speed

# --- Test Mode Toggle ---
TEST_MODE = False  # Set to False for full experiments

if TEST_MODE:
    SCALABILITY_TEST = True
    SCALABILITY_AGENT_COUNTS = [2, 5]  
    SCALABILITY_META_ITERS = 20
    SCALABILITY_EPISODES = 5
    SEEDS = [42]  
    META_ITERATIONS = 50
    EPISODES = 10
    MAX_AGENTS = 5  # ← ADD THIS - Keep at 5 for test mode
else:
    # Full experimental settings
    SCALABILITY_TEST = False
    SCALABILITY_AGENT_COUNTS = [2, 5, 8, 12, 16]
    SCALABILITY_META_ITERS = 100
    SCALABILITY_EPISODES = 10
    SEEDS = [42, 123, 888, 2024, 7, 9999, 3141, 1618, 2718, 404]
    META_ITERATIONS = 500  # ← ADD THIS
    EPISODES = 20  # ← ADD THIS
    MAX_AGENTS = 16  # ← IMPORTANT: Increase for scalability to 16 agents
    print("\n" + "="*60)
    print("RUNNING FULL EXPERIMENT MODE")
    print("="*60)
    print(f"Settings:")
    print(f"  - Seeds: {len(SEEDS)} seeds")
    print(f"  - Scalability agent counts: {SCALABILITY_AGENT_COUNTS}")
    print(f"  - Meta iterations: {META_ITERATIONS}")
    print(f"  - Max agents: {MAX_AGENTS}")
    print(f"  - Estimated time: 8-15 hours")
    print("="*60 + "\n")


# --- LoRASA Layer ---
class LoRASALayer(nn.Module):
    def __init__(self, linear_layer, rank=4):
        super().__init__()
        self.linear = linear_layer
        in_features, out_features = linear_layer.in_features, linear_layer.out_features
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None: self.linear.bias.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_scale = nn.Parameter(torch.ones(out_features))
        self.lora_shift = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
    def forward(self, x):
        frozen_output = self.linear(x)
        adapter_output = (x @ self.lora_A @ self.lora_B) * self.lora_scale + self.lora_shift
        return frozen_output + adapter_output
    
    
# --- Profiling & Measurement Utilities ---
import psutil
import os

class PerformanceProfiler:
    """Tracks memory, time, and computational metrics during training"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.peak_memory_cpu = 0
        self.peak_memory_gpu = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def update(self):
        """Update peak memory measurements"""
        # CPU memory in MB
        process = psutil.Process(os.getpid())
        current_cpu_mem = process.memory_info().rss / 1024 / 1024
        self.peak_memory_cpu = max(self.peak_memory_cpu, current_cpu_mem)
        
        # GPU memory in MB
        if torch.cuda.is_available():
            current_gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            self.peak_memory_gpu = max(self.peak_memory_gpu, current_gpu_mem)
    
    def get_metrics(self):
        """Get all collected metrics"""
        elapsed = time.time() - self.start_time
        return {
            'time_seconds': elapsed,
            'cpu_memory_mb': self.peak_memory_cpu,
            'gpu_memory_mb': self.peak_memory_gpu if torch.cuda.is_available() else 0
        }

def measure_inference_latency(model, env_fn, num_agents, num_samples=100):
    """Measure average inference time per forward pass"""
    env = env_fn(num_agents=num_agents)
    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    
    # Determine model type and appropriate forward method
    is_mappo = isinstance(model, MAPPOPolicy)
    is_maml = isinstance(model, MAMLPolicy)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(10):
            if is_mappo:
                _ = model.get_action(obs_tensor.squeeze(0), num_agents)
            else:
                _ = model(obs_tensor.squeeze(0) if is_maml else obs_tensor)
    
    # Actual measurement
    latencies = []
    with torch.no_grad():
        for _ in range(num_samples):
            start = time.perf_counter()
            if is_mappo:
                _ = model.get_action(obs_tensor.squeeze(0), num_agents)
            else:
                _ = model(obs_tensor.squeeze(0) if is_maml else obs_tensor)
            latencies.append((time.perf_counter() - start) * 1000)  # Convert to ms
    
    return np.mean(latencies), np.std(latencies)


def measure_ippo_latency(ippo_policies, env_fn, num_agents, num_samples=100):
    """Measure IPPO inference latency (policy forward pass only)"""
    
    # Check if policies exist for this agent count
    if num_agents not in ippo_policies:
        return 0.0, 0.0  # Return zero if no policy trained
    
    # Get observation space dimensions
    env = env_fn(num_agents=num_agents)
    obs, _ = env.reset()
    
    policy = ippo_policies[num_agents]
    
    # Warmup runs
    for _ in range(10):
        _, _ = policy.predict(obs, deterministic=True)
    
    # Actual measurement
    latencies = []
    for _ in range(num_samples):
        start = time.perf_counter()
        _, _ = policy.predict(obs, deterministic=True)
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    
    return np.mean(latencies), np.std(latencies)

def count_attention_complexity(num_agents):
    """Calculate theoretical attention complexity O(N^2)"""
    return num_agents * num_agents


# --- Environments ---
class CustomMultiAgentEnv(gym.Env):
    def __init__(self, num_agents=2, seed=None):
        super().__init__()
        self.num_agents = num_agents
        self.observation_space = spaces.Box(low=-1, high=1, shape=(MAX_AGENTS * 4,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2] * MAX_AGENTS)
        self.max_steps = 200; self.steps = 0; self.seed = seed
        self.randomize_environment()
    def randomize_environment(self):
        local_random_state = np.random.RandomState(self.seed)
        self.move_range = local_random_state.randint(3, 20)
        self.start_bounds = local_random_state.randint(20, 150)
        self.agent_positions = local_random_state.randint(self.start_bounds, 600 - self.start_bounds, (self.num_agents, 2))
        self.agent_speeds = local_random_state.uniform(0.5, 2.0, size=self.num_agents)
        self.goal_zones = local_random_state.randint(50, 550, size=(self.num_agents, 2))
        self.agent_states = np.random.uniform(low=-1, high=1, size=(self.num_agents, 4))
        self.prev_distances = np.array([np.linalg.norm(self.agent_positions[i] - self.goal_zones[i]) for i in range(self.num_agents)])
    def reset(self, seed=None, options=None):
        super().reset(seed=seed if seed is not None else self.seed)
        self.randomize_environment(); self.steps = 0
        obs = self.agent_states.flatten()
        padded_obs = np.zeros((MAX_AGENTS * 4,), dtype=np.float32); padded_obs[:len(obs)] = obs
        return padded_obs, {}
    def step(self, actions):
        actions = np.array(actions).flatten()[:self.num_agents]
        rewards = np.zeros(self.num_agents)
        move_mask = (actions == 1)
        if np.sum(move_mask) > 0:
            move_amounts = np.random.randint(-self.move_range, self.move_range, size=(np.sum(move_mask), 2))
            for i, idx in enumerate(np.where(move_mask)[0]):
                move_amounts[i] = (move_amounts[i] * self.agent_speeds[idx]).astype(int)
            self.agent_positions[move_mask] += move_amounts
        self.agent_positions = np.clip(self.agent_positions, self.start_bounds, 600 - self.start_bounds)
        new_distances = np.array([np.linalg.norm(self.agent_positions[i] - self.goal_zones[i]) for i in range(self.num_agents)])
        for i in range(self.num_agents):
            rewards[i] += max(0, 100 - new_distances[i]) / 100
            rewards[i] += 0.5 * max(0, (self.prev_distances[i] - new_distances[i]) / 100)
        self.prev_distances = new_distances
        self.steps += 1; terminated = self.steps >= self.max_steps
        noisy_obs = self.agent_states + np.random.normal(0, 0.2, size=self.agent_states.shape)
        obs = noisy_obs.flatten()
        padded_obs = np.zeros((MAX_AGENTS * 4,), dtype=np.float32); padded_obs[:len(obs)] = obs
        return padded_obs, np.sum(rewards) / self.num_agents, terminated, False, {}

class UnseenCustomMultiAgentEnv(CustomMultiAgentEnv):
    def randomize_environment(self):
        super().randomize_environment()
        local_random_state = np.random.RandomState(self.seed)
        self.move_range = local_random_state.randint(25, 50)
        self.observation_noise_std = 0.4
    def step(self, actions):
        padded_obs, reward, terminated, truncated, info = super().step(actions)
        centroid = np.mean(self.agent_positions, axis=0)
        cohesion_penalty = -0.1 * np.mean(np.linalg.norm(self.agent_positions - centroid, axis=1)) / 100.0
        return padded_obs, reward + cohesion_penalty, terminated, truncated, info

class SimpleSpreadWrapper(gym.Env):
    @staticmethod
    def get_max_obs_dim(max_agents):
        temp_env = simple_spread_v3.parallel_env(N=max_agents)
        max_dim = temp_env.observation_space("agent_0").shape[0]
        temp_env.close()
        return max_dim
    
    def __init__(self, num_agents=5, seed=None):
        self.env = simple_spread_v3.parallel_env(N=num_agents, max_cycles=100, continuous_actions=False)
        self.num_agents = num_agents
        self.seed = seed
        self.max_obs_dim_per_agent = self.get_max_obs_dim(MAX_AGENTS)
        self.action_dim = self.env.action_space("agent_0").n
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(MAX_AGENTS * self.max_obs_dim_per_agent,), 
            dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([self.action_dim] * MAX_AGENTS)
        
        # Only include actual agents, not MAX_AGENTS
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
                
        self.reset(seed=self.seed)
    
    def reset(self, seed=None, options=None):
        obs_dict, infos = self.env.reset(seed=seed if seed is not None else self.seed)
        return self.flatten_obs(obs_dict), infos
    
    def step(self, actions):
        
        # Handle actions array - trim to actual number of agents
        if isinstance(actions, np.ndarray):
            actions = actions.flatten()[:self.num_agents]
        elif isinstance(actions, torch.Tensor):
            actions = actions.numpy().flatten()[:self.num_agents]
        
        # Create action dict only for actual agents
        action_dict = {agent: actions[i] for i, agent in enumerate(self.agents)}
        
        next_obs_dict, rewards_dict, terminateds_dict, truncateds_dict, infos = self.env.step(action_dict)
        
        reward = sum(rewards_dict.values()) / self.num_agents if self.num_agents > 0 else 0
        terminated = all(terminateds_dict.values())
        
        return self.flatten_obs(next_obs_dict), reward, terminated, False, {}
    
    def flatten_obs(self, obs_dict):
        padded_obs = np.zeros((MAX_AGENTS, self.max_obs_dim_per_agent), dtype=np.float32)
        
        for i, agent_name in enumerate(self.agents):
            if agent_name in obs_dict:
                obs = obs_dict[agent_name]
                padded_obs[i, :len(obs)] = obs
        
        return padded_obs.flatten()


# --- Models ---
class LightMetaPolicy(nn.Module):
    """
    The definitive champion LightMetaPolicy architecture based on comprehensive tuning.
    It uses a residual connection but disables Layer Normalization and the relational
    bias for the best all-around performance and efficiency.
    """
    def __init__(self, agent_obs_dim, num_actions, use_layer_norm=False, use_residual=True, use_relational_bias=False):
        super().__init__()
        self.agent_dim, self.num_actions = agent_obs_dim, num_actions
        self.use_layer_norm, self.use_residual = use_layer_norm, use_residual
        self.use_relational_bias = use_relational_bias
        self.d_model = 64
        
        self.input_proj = nn.Linear(self.agent_dim, self.d_model)
        self.key_transform = nn.Linear(self.d_model, self.d_model)
        self.query_transform = nn.Linear(self.d_model, self.d_model)
        self.value_transform = nn.Linear(self.d_model, self.d_model)
        
        # These layers are conditionally created but will be unused with default champion settings
        if self.use_relational_bias: self.agent_relation = nn.Linear(self.d_model, self.d_model)
        if self.use_layer_norm: self.layer_norm1 = nn.LayerNorm(self.d_model)
        
        self.fc_out = nn.Linear(self.d_model, self.d_model)
        self.ffn = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.GELU(), nn.Linear(self.d_model, self.d_model))
        
        if self.use_layer_norm: self.layer_norm2 = nn.LayerNorm(self.d_model)
        
        self.action_head = nn.Linear(self.d_model, MAX_AGENTS * self.num_actions)
        self.value_head = nn.Sequential(nn.Linear(self.d_model, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        agents = x.view(batch_size, MAX_AGENTS, self.agent_dim)
        
        projected_agents = self.input_proj(agents)
        
        queries = self.query_transform(projected_agents).unsqueeze(2)
        keys = self.key_transform(projected_agents).unsqueeze(2)
        values = self.value_transform(projected_agents).unsqueeze(2)
        
        attention = torch.matmul(queries, keys.transpose(-2, -1)) / (self.d_model ** 0.5)
        
        # This block will be skipped with the default champion settings
        if self.use_relational_bias:
            rel_queries = queries.squeeze(2)
            rel_bias_logits = self.agent_relation(rel_queries).unsqueeze(2)
            attention += torch.matmul(rel_bias_logits, queries.transpose(-2,-1)) / (self.d_model ** 0.5)
            
        attention = torch.softmax(attention, dim=-1)
        context = torch.matmul(attention, values).squeeze(2)
        context = self.fc_out(context)
        
        if self.use_residual: 
            context = context + projected_agents
        # This block will be skipped with the default champion settings
        if self.use_layer_norm: 
            context = self.layer_norm1(context)
            
        agent_mask = (agents.abs().sum(dim=-1, keepdim=True) > 0.01).float()
        pooled = (context * agent_mask).sum(dim=1) / (agent_mask.sum(dim=1) + 1e-8)
        
        ffn_out = self.ffn(pooled)
        if self.use_residual: 
            ffn_out = ffn_out + pooled
        # This block will be skipped with the default champion settings
        if self.use_layer_norm: 
            ffn_out = self.layer_norm2(ffn_out)
            
        action_logits = self.action_head(ffn_out).view(batch_size, MAX_AGENTS, self.num_actions)
        value = self.value_head(pooled)
        return action_logits, value


class MAPPOPolicy(nn.Module):
    def __init__(self, agent_obs_dim, num_actions):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(agent_obs_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_actions))
        self.critic = nn.Sequential(nn.Linear(agent_obs_dim * MAX_AGENTS, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def get_action(self, obs, num_agents, deterministic=False):
        agent_obs = obs.view(MAX_AGENTS, -1)[:num_agents]
        dist = torch.distributions.Categorical(logits=self.actor(agent_obs))
        return dist.sample() if not deterministic else torch.argmax(dist.logits, dim=-1)
    def get_value(self, all_obs): return self.critic(all_obs)

class MAMLPolicy(nn.Module):
    def __init__(self, input_dim, output_logit_dim):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(input_dim, 128), nn.Linear(128, 128), nn.Linear(128, output_logit_dim)
    def forward(self, x): return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
    def adapt(self, loss, lr=0.01):
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
        with torch.no_grad():
            for param, grad in zip(self.parameters(), grads): param -= lr * grad
        return self

def inject_lorasa(model, rank=4):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRASALayer(module, rank=rank))
        elif len(list(module.children())) > 0:
            inject_lorasa(module, rank=rank)
    return model

def freeze_base_parameters(model):
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

# --- Training & Fine-Tuning Functions ---
def train_light_meta_policy(model, env_fn, meta_iterations=400, inner_rollouts=128, n_epochs=4, gamma=0.99, entropy_coef=0.01, verbose=False, profiler=None, fixed_num_agents=None):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    start_time = time.time()
    learning_curve = []
    total_steps = 0
    
    for iteration in range(meta_iterations):
        # Update profiler every 50 iterations
        if profiler and iteration % 50 == 0:
            profiler.update()
        
        # Use fixed_num_agents for scalability, or random for normal training
        if fixed_num_agents is not None:
            num_agents = fixed_num_agents
        else:
            num_agents = np.random.choice([2, 3, 4, 5])
        
        env = env_fn(num_agents=num_agents)
        obs, _ = env.reset()
        all_obs, all_rewards, all_actions = [], [], []
        
        for _ in range(inner_rollouts):
            with torch.no_grad():
                action_logits, _ = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                dist = torch.distributions.Categorical(logits=action_logits[:, :num_agents, :])
                actions = dist.sample()
            next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
            all_obs.append(obs)
            all_actions.append(actions.numpy())
            all_rewards.append(reward)
            obs = next_obs
            total_steps += 1
            if terminated:
                obs, _ = env.reset()
        
        obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32)
        actions_tensor = torch.tensor(np.array(all_actions), dtype=torch.long)
        
        returns = []
        discounted_reward = 0
        for r in reversed(all_rewards):
            discounted_reward = r + gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        for _ in range(n_epochs):
            action_logits, value_pred = model(obs_tensor)
            dist = torch.distributions.Categorical(logits=action_logits[:, :num_agents, :])
            new_log_probs = dist.log_prob(actions_tensor)
            advantages = returns - value_pred.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            policy_loss = -(new_log_probs.mean(dim=-1) * advantages).mean()
            value_loss = nn.MSELoss()(value_pred.squeeze(), returns)
            loss = policy_loss + 0.5 * value_loss - entropy_coef * dist.entropy().mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        
        if iteration % 20 == 0:
            eval_reward, _, _ = evaluate_policy(model, env_fn)
            learning_curve.append((total_steps, eval_reward))
    
    return model, time.time() - start_time, learning_curve

def train_mappo_policy(model, env_fn, iterations=1000, rollout_steps=512, n_epochs=4, gamma=0.99, verbose=False, profiler=None, fixed_num_agents=None):
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    start_time = time.time()
    learning_curve = []
    total_steps = 0
    
    for iteration in range(iterations):
        # Update profiler every 50 iterations
        if profiler and iteration % 50 == 0:
            profiler.update()
        
        # Use fixed_num_agents for scalability, or random for normal training
        if fixed_num_agents is not None:
            num_agents = fixed_num_agents
        else:
            num_agents = np.random.choice([2, 3, 4, 5])
        
        env = env_fn(num_agents=num_agents)
        obs, _ = env.reset()
        all_obs, all_rewards, all_actions = [], [], []
        
        for _ in range(rollout_steps):
            with torch.no_grad():
                actions = model.get_action(torch.tensor(obs, dtype=torch.float32), num_agents)
            next_obs, reward, terminated, _, _ = env.step(actions.numpy())
            all_obs.append(obs)
            all_actions.append(actions.numpy())
            all_rewards.append(reward)
            obs = next_obs
            total_steps += 1
            if terminated:
                obs, _ = env.reset()
        
        obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32)
        actions_tensor = torch.tensor(np.array(all_actions), dtype=torch.long)
        
        returns = []
        discounted_reward = 0
        for r in reversed(all_rewards):
            discounted_reward = r + gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        for _ in range(n_epochs):
            dist = torch.distributions.Categorical(logits=model.actor(obs_tensor.view(-1, MAX_AGENTS, model.actor[0].in_features)[:, :num_agents, :]))
            new_log_probs = dist.log_prob(actions_tensor)
            value_pred = model.get_value(obs_tensor)
            advantages = returns - value_pred.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            policy_loss = -(new_log_probs.mean(dim=-1) * advantages).mean()
            value_loss = nn.MSELoss()(value_pred.squeeze(), returns)
            loss = policy_loss + value_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if iteration % 50 == 0:
            eval_reward, _, _ = evaluate_policy(model, env_fn, is_mappo=True)
            learning_curve.append((total_steps, eval_reward))
    
    return model, time.time() - start_time, learning_curve


def train_ippo_policy(env_fn, timesteps=100000, profiler=None, fixed_num_agents=None):
    """
    Train Independent PPO policies for multi-agent environments.
    
    Args:
        env_fn: Function to create environment
        timesteps: Total training timesteps
        profiler: Optional performance profiler
        fixed_num_agents: If provided, only train for this specific agent count (for scalability)
    """
    policies = {}
    start_time = time.time()
    
    # Determine which agent counts to train for
    if fixed_num_agents is not None:
        # Scalability mode - train only for specific agent count
        agent_counts = [fixed_num_agents]
        print(f"  IPPO: Training for {fixed_num_agents} agents only (scalability mode)")
    else:
        # Normal mode - train for all agent counts
        agent_counts = [2, 3, 4, 5]
        print(f"  IPPO: Training for agent counts {agent_counts}")
    
    for num_agents in agent_counts:
        # Update profiler before training each agent count
        if profiler:
            profiler.update()
        
        env = env_fn(num_agents=num_agents)
        policies[num_agents] = PPO(
            'MlpPolicy', 
            env, 
            verbose=0, 
            n_steps=2048, 
            learning_rate=0.0003, 
            device='cpu'
        )
        policies[num_agents].learn(total_timesteps=timesteps // len(agent_counts))
        print(f"    ✓ IPPO trained for {num_agents} agents")
    
    # Final profiler update
    if profiler:
        profiler.update()
    
    return policies, time.time() - start_time



def meta_train_maml(model, env_fn, meta_iterations=750, inner_steps=10, inner_rollouts=50, gamma=0.99, inner_lr=0.01, verbose=False, profiler=None, fixed_num_agents=None):
    meta_optimizer = optim.Adam(model.parameters(), lr=0.0003)
    start_time = time.time()
    
    for iteration in range(meta_iterations):
        # Update profiler every 50 iterations
        if profiler and iteration % 50 == 0:
            profiler.update()
        
        # Use fixed_num_agents for scalability, or random for normal training
        if fixed_num_agents is not None:
            num_agents = fixed_num_agents
        else:
            num_agents = np.random.choice([2, 3, 4, 5])
        
        env = env_fn(num_agents=num_agents)
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        
        adapted_model = copy.deepcopy(model)
        for _ in range(inner_steps):
            action_logits = adapted_model(obs)
            dist = torch.distributions.Categorical(logits=action_logits.view(1, MAX_AGENTS, -1)[:, :num_agents, :])
            actions = dist.sample()
            next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
            loss = -dist.log_prob(actions).mean() * reward
            adapted_model = adapted_model.adapt(loss, lr=inner_lr)
            obs = torch.tensor(next_obs, dtype=torch.float32)
            if terminated:
                obs, _ = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32)
        
        log_probs, rewards = [], []
        for _ in range(inner_rollouts):
            action_logits = adapted_model(obs)
            dist = torch.distributions.Categorical(logits=action_logits.view(1, MAX_AGENTS, -1)[:, :num_agents, :])
            actions = dist.sample()
            next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
            log_probs.append(dist.log_prob(actions).mean())
            rewards.append(reward)
            obs = torch.tensor(next_obs, dtype=torch.float32)
            if terminated:
                break
        
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        loss = -torch.sum(torch.stack(log_probs) * torch.tensor(returns, dtype=torch.float32))
        meta_optimizer.zero_grad()
        loss.backward()
        meta_optimizer.step()
    
    return model, time.time() - start_time

def fine_tune_standard(model, env_fn, max_steps=40):
    tuned_model = copy.deepcopy(model)
    optimizer = optim.Adam(tuned_model.parameters(), lr=0.0001)
    env = env_fn()
    experiences = []
    for _ in range(5):
        obs, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action_logits, _ = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                dist = torch.distributions.Categorical(logits=action_logits[:, :env.num_agents, :])
                actions = dist.sample()
            next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
            experiences.append((obs, actions.numpy().flatten(), reward, next_obs))
            obs = next_obs; done = terminated
    for _ in range(max_steps):
        if len(experiences) > 64:
            batch = random.sample(experiences, 64)
            batch_obs = torch.tensor(np.array([s[0] for s in batch]), dtype=torch.float32)
            batch_actions = torch.tensor(np.array([s[1] for s in batch]), dtype=torch.long)
            batch_rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32)
            action_logits, _ = tuned_model(batch_obs)
            dist = torch.distributions.Categorical(logits=action_logits[:, :env.num_agents, :])
            log_probs = dist.log_prob(batch_actions)
            loss = - (log_probs * batch_rewards.unsqueeze(1)).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return tuned_model

def fine_tune_with_lorasa(model, env_fn, max_steps=40):
    tuned_model = copy.deepcopy(model)
    inject_lorasa(tuned_model)
    freeze_base_parameters(tuned_model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, tuned_model.parameters()), lr=0.001)
    env = env_fn()
    experiences = []
    for _ in range(5):
        obs, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action_logits, _ = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                dist = torch.distributions.Categorical(logits=action_logits[:, :env.num_agents, :])
                actions = dist.sample()
            next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
            experiences.append((obs, actions.numpy().flatten(), reward, next_obs))
            obs = next_obs; done = terminated
    for _ in range(max_steps):
        if len(experiences) > 64:
            batch = random.sample(experiences, 64)
            batch_obs = torch.tensor(np.array([s[0] for s in batch]), dtype=torch.float32)
            batch_actions = torch.tensor(np.array([s[1] for s in batch]), dtype=torch.long)
            batch_rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32)
            action_logits, _ = tuned_model(batch_obs)
            dist = torch.distributions.Categorical(logits=action_logits[:, :env.num_agents, :])
            log_probs = dist.log_prob(batch_actions)
            loss = - (log_probs * batch_rewards.unsqueeze(1)).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return tuned_model

def fine_tune_maml_policy(meta_model, env_fn, steps=40):
    env = env_fn()
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    adapted_model = copy.deepcopy(meta_model)
    for _ in range(steps):
        action_logits = adapted_model(obs)
        dist = torch.distributions.Categorical(logits=action_logits.view(1, MAX_AGENTS, -1)[:,:env.num_agents,:])
        actions = dist.sample()
        next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
        loss = -dist.log_prob(actions).mean() * reward
        adapted_model = adapted_model.adapt(loss)
        obs = torch.tensor(next_obs, dtype=torch.float32)
        if terminated: obs, _ = env.reset(); obs = torch.tensor(obs, dtype=torch.float32)
    return adapted_model

def evaluate_policy(model, env_fn, episodes=EPISODES, is_mappo=False, is_ippo=False, ippo_policies=None):
    rewards = []
    for ep in range(episodes):
        env = env_fn(seed=1000 + ep)
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                if is_mappo:
                    actions = model.get_action(torch.tensor(obs, dtype=torch.float32), env.num_agents, deterministic=True).numpy()
                elif is_ippo:
                    # Safety check: ensure policy exists for this agent count
                    if env.num_agents not in ippo_policies:
                        print(f"⚠️  Warning: IPPO policy not available for {env.num_agents} agents. Returning default poor reward.")
                        return -10000.0, 0.0, [-10000.0] * episodes
                    actions, _ = ippo_policies[env.num_agents].predict(obs, deterministic=True)
                else: 
                    output = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                    action_logits = output[0] if isinstance(output, tuple) else output
                    if action_logits.dim() == 2:
                        num_actions = action_logits.shape[1] // MAX_AGENTS
                        action_logits = action_logits.view(1, MAX_AGENTS, num_actions)
                    actions = torch.argmax(action_logits[:, :env.num_agents, :], dim=-1).numpy().flatten()
            
            obs, reward, terminated, _, _ = env.step(actions)
            total_reward += reward
            done = terminated
        
        rewards.append(total_reward)
    
    return np.mean(rewards), np.std(rewards), rewards


def run_scalability_test(seed, env_config):
    """Test all models' performance across different agent counts"""
    global MAX_AGENTS
    
    print(f"\n{'='*60}")
    print(f"SCALABILITY TEST - Seed: {seed}")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    results = {
        'agent_counts': SCALABILITY_AGENT_COUNTS,
        'LightMeta': {'rewards': [], 'times': [], 'memory': [], 'latency': []},
        'MAPPO': {'rewards': [], 'times': [], 'memory': [], 'latency': []},
        'IPPO': {'rewards': [], 'times': [], 'memory': [], 'latency': []},
        'MAML': {'rewards': [], 'times': [], 'memory': [], 'latency': []}
    }
    
    env_fn = env_config['fn']
    agent_obs_dim = env_config['agent_obs_dim']
    num_actions = env_config['num_actions']
    input_dim = MAX_AGENTS * agent_obs_dim
    
    for num_agents in SCALABILITY_AGENT_COUNTS:
        print(f"\n  → Testing with {num_agents} agents...")
        
        # Temporarily adjust MAX_AGENTS for larger scales
        original_max = MAX_AGENTS
        MAX_AGENTS = max(num_agents, original_max)
        
        # Create wrapper function that provides consistent agent count
        def make_eval_env_fn(fixed_num_agents, base_env_fn):
            """Wrapper to handle num_agents parameter from training functions"""
            def env_wrapper(num_agents=None, seed=None):
                # ALWAYS use fixed_num_agents for scalability
                kwargs = {'num_agents': fixed_num_agents}
                if seed is not None:
                    kwargs['seed'] = seed
                return base_env_fn(**kwargs)
            return env_wrapper

        eval_env_fn = make_eval_env_fn(num_agents, env_fn)
        
        # --- LightMeta ---
        profiler_lm = PerformanceProfiler()
        model_lm = LightMetaPolicy(agent_obs_dim, num_actions)
        model_lm, train_time_lm, _ = train_light_meta_policy(
            model_lm, eval_env_fn, 
            meta_iterations=SCALABILITY_META_ITERS,
            verbose=False,
            profiler=profiler_lm,
            fixed_num_agents=num_agents  # ← ADD THIS
        )
        metrics_lm = profiler_lm.get_metrics()
        reward_lm, _, _ = evaluate_policy(model_lm, eval_env_fn, episodes=SCALABILITY_EPISODES)
        latency_lm, _ = measure_inference_latency(model_lm, eval_env_fn, num_agents)
        
        results['LightMeta']['rewards'].append(reward_lm)
        results['LightMeta']['times'].append(metrics_lm['time_seconds'])
        results['LightMeta']['memory'].append(metrics_lm['cpu_memory_mb'])
        results['LightMeta']['latency'].append(latency_lm)
        
        # --- MAPPO ---
        profiler_mappo = PerformanceProfiler()
        model_mappo = MAPPOPolicy(agent_obs_dim, num_actions)
        model_mappo, train_time_mappo, _ = train_mappo_policy(
            model_mappo, eval_env_fn,
            iterations=SCALABILITY_META_ITERS * 2,
            verbose=False,
            profiler=profiler_mappo,
            fixed_num_agents=num_agents  # ← ADD THIS
        )
        metrics_mappo = profiler_mappo.get_metrics()
        reward_mappo, _, _ = evaluate_policy(model_mappo, eval_env_fn, is_mappo=True, episodes=SCALABILITY_EPISODES)
        latency_mappo, _ = measure_inference_latency(model_mappo, eval_env_fn, num_agents)
        
        results['MAPPO']['rewards'].append(reward_mappo)
        results['MAPPO']['times'].append(metrics_mappo['time_seconds'])
        results['MAPPO']['memory'].append(metrics_mappo['cpu_memory_mb'])
        results['MAPPO']['latency'].append(latency_mappo)
        
        # --- IPPO ---
        profiler_ippo = PerformanceProfiler()
        ippo_policies, train_time_ippo = train_ippo_policy(
            eval_env_fn, 
            timesteps=25000,
            profiler=profiler_ippo,
            fixed_num_agents=num_agents  # Train only for current agent count
        )
        metrics_ippo = profiler_ippo.get_metrics()
        reward_ippo, _, _ = evaluate_policy(None, eval_env_fn, is_ippo=True, 
                                        ippo_policies=ippo_policies, episodes=SCALABILITY_EPISODES)
        latency_ippo, _ = measure_ippo_latency(ippo_policies, eval_env_fn, num_agents)

        results['IPPO']['rewards'].append(reward_ippo)
        results['IPPO']['times'].append(train_time_ippo)
        results['IPPO']['memory'].append(metrics_ippo['cpu_memory_mb'])
        results['IPPO']['latency'].append(latency_ippo)

        # --- MAML ---
        profiler_maml = PerformanceProfiler()
        model_maml = MAMLPolicy(input_dim, MAX_AGENTS * num_actions)
        model_maml, train_time_maml = meta_train_maml(
            model_maml, eval_env_fn, 
            meta_iterations=int(SCALABILITY_META_ITERS * 1.5),
            profiler=profiler_maml,
            fixed_num_agents=num_agents  # ← ADD THIS
        )
        metrics_maml = profiler_maml.get_metrics()
        reward_maml, _, _ = evaluate_policy(model_maml, eval_env_fn, episodes=SCALABILITY_EPISODES)
        latency_maml, _ = measure_inference_latency(model_maml, eval_env_fn, num_agents)
        
        results['MAML']['rewards'].append(reward_maml)
        results['MAML']['times'].append(train_time_maml)
        results['MAML']['memory'].append(metrics_maml['cpu_memory_mb'])
        results['MAML']['latency'].append(latency_maml)
        
        # Restore MAX_AGENTS
        MAX_AGENTS = original_max
        
        print(f"    ✓ Completed {num_agents} agents")
    
    return results




def plot_scalability_results(all_scalability_results):
    """Generate comprehensive scalability visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Scalability Analysis: All Methods Comparison', fontsize=18, fontweight='bold')
    
    agent_counts = all_scalability_results[0]['agent_counts']
    
    # Define visual styles
    method_styles = {
        'LightMeta': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'linewidth': 2.5},
        'MAPPO': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--', 'linewidth': 2.5},
        'IPPO': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.', 'linewidth': 2.5},
        'MAML': {'color': '#d62728', 'marker': 'd', 'linestyle': ':', 'linewidth': 2.5}
    }
    
    # Aggregate data across seeds
    for model_name in ['LightMeta', 'MAPPO', 'IPPO', 'MAML']:
        rewards = np.array([r[model_name]['rewards'] for r in all_scalability_results])
        times = np.array([r[model_name]['times'] for r in all_scalability_results])
        memory = np.array([r[model_name]['memory'] for r in all_scalability_results])
        latency = np.array([r[model_name]['latency'] for r in all_scalability_results])
        
        style = method_styles[model_name]
        
        # Plot 1: Performance vs Agent Count
        axes[0, 0].plot(agent_counts, rewards.mean(axis=0), 
                       label=model_name, marker=style['marker'],
                       color=style['color'], linestyle=style['linestyle'],
                       linewidth=style['linewidth'], markersize=8)
        axes[0, 0].fill_between(agent_counts, 
                                rewards.mean(axis=0) - rewards.std(axis=0),
                                rewards.mean(axis=0) + rewards.std(axis=0),
                                alpha=0.15, color=style['color'])
        
        # Plot 2: Training Time vs Agent Count
        axes[0, 1].plot(agent_counts, times.mean(axis=0), 
                       label=model_name, marker=style['marker'],
                       color=style['color'], linestyle=style['linestyle'],
                       linewidth=style['linewidth'], markersize=8)
        axes[0, 1].fill_between(agent_counts,
                                times.mean(axis=0) - times.std(axis=0),
                                times.mean(axis=0) + times.std(axis=0),
                                alpha=0.15, color=style['color'])
        
        # Plot 3: Memory Consumption vs Agent Count
        axes[1, 0].plot(agent_counts, memory.mean(axis=0), 
                       label=model_name, marker=style['marker'],
                       color=style['color'], linestyle=style['linestyle'],
                       linewidth=style['linewidth'], markersize=8)
        axes[1, 0].fill_between(agent_counts,
                                memory.mean(axis=0) - memory.std(axis=0),
                                memory.mean(axis=0) + memory.std(axis=0),
                                alpha=0.15, color=style['color'])
        
        # Plot 4: Inference Latency vs Agent Count
        axes[1, 1].plot(agent_counts, latency.mean(axis=0), 
                       label=model_name, marker=style['marker'],
                       color=style['color'], linestyle=style['linestyle'],
                       linewidth=style['linewidth'], markersize=8)
        axes[1, 1].fill_between(agent_counts,
                                latency.mean(axis=0) - latency.std(axis=0),
                                latency.mean(axis=0) + latency.std(axis=0),
                                alpha=0.15, color=style['color'])
    
    # Formatting
    axes[0, 0].set_xlabel('Number of Agents', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Average Return', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Performance Scaling', fontsize=14)
    axes[0, 0].legend(fontsize=11, loc='best', framealpha=0.9)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    axes[0, 1].set_xlabel('Number of Agents', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Training Time Scaling', fontsize=14)
    axes[0, 1].legend(fontsize=11, loc='best', framealpha=0.9)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    axes[1, 0].set_xlabel('Number of Agents', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Peak CPU Memory (MB)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Memory Consumption', fontsize=14)
    axes[1, 0].legend(fontsize=11, loc='best', framealpha=0.9)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    axes[1, 1].set_xlabel('Number of Agents', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Inference Latency (ms)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Inference Speed', fontsize=14)
    axes[1, 1].legend(fontsize=11, loc='best', framealpha=0.9)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('scalability_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Scalability plot saved as 'scalability_analysis.png'")
    plt.show()
    
    # Print detailed table
    print("\n" + "="*90)
    print("SCALABILITY METRICS SUMMARY (Averaged Across Seeds)")
    print("="*90)
    print(f"{'Method':<12} | {'Agents':<8} | {'Return':<18} | {'Time (s)':<18} | {'Memory (MB)':<18}")
    print("-"*90)
    
    for model_name in ['LightMeta', 'MAPPO', 'IPPO', 'MAML']:
        for i, n in enumerate(agent_counts):
            rewards = [r[model_name]['rewards'][i] for r in all_scalability_results]
            times = [r[model_name]['times'][i] for r in all_scalability_results]
            memory = [r[model_name]['memory'][i] for r in all_scalability_results]
            
            print(f"{model_name:<12} | {n:<8} | "
                  f"{np.mean(rewards):7.2f} ± {np.std(rewards):5.2f}  | "
                  f"{np.mean(times):7.2f} ± {np.std(times):5.2f}  | "
                  f"{np.mean(memory):7.1f} ± {np.std(memory):5.1f}")
    
    # O(N^2) complexity analysis
    print("\n" + "="*70)
    print("ATTENTION COMPLEXITY ANALYSIS: O(N²) vs O(N)")
    print("="*70)
    print(f"{'Agent Count (N)':<20} | {'Attention Ops (N²)':<25} | {'Growth Factor':<20}")
    print("-"*70)
    
    prev_ops = None
    for n in agent_counts:
        ops = count_attention_complexity(n)
        growth = f"{ops/prev_ops:.2f}x" if prev_ops else "baseline"
        print(f"{n:<20} | {ops:<25} | {growth:<20}")
        prev_ops = ops
    
    # Statistical significance for largest scale
    print("\n" + "="*70)
    print(f"STATISTICAL SIGNIFICANCE AT N={agent_counts[-1]} AGENTS")
    print("="*70)
    
    from scipy.stats import ttest_ind
    
    largest_idx = -1
    lm_rewards = [r['LightMeta']['rewards'][largest_idx] for r in all_scalability_results]
    mappo_rewards = [r['MAPPO']['rewards'][largest_idx] for r in all_scalability_results]
    ippo_rewards = [r['IPPO']['rewards'][largest_idx] for r in all_scalability_results]
    maml_rewards = [r['MAML']['rewards'][largest_idx] for r in all_scalability_results]
    
    comparisons = [
        ('LightMeta', 'MAPPO', lm_rewards, mappo_rewards),
        ('LightMeta', 'IPPO', lm_rewards, ippo_rewards),
        ('LightMeta', 'MAML', lm_rewards, maml_rewards)
    ]
    
    for method1, method2, data1, data2 in comparisons:
        t_stat, p_val = ttest_ind(data1, data2)
        sig = "✓ SIGNIFICANT" if p_val < 0.05 else "✗ Not significant"
        print(f"  {method1} vs {method2}: t={t_stat:.3f}, p={p_val:.4f} {sig}")



def export_results_to_csv(all_runs_results, scalability_results):
    """Export all experimental results to CSV files for paper inclusion"""
    import csv
    
    # Export main comparison results
    with open('main_comparison_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Variant', 'N_Agents', 'Mean_Reward', 'Std_Reward', 'CI_95_Lower', 'CI_95_Upper', 'N_Seeds'])
        
        # LightMeta variants
        for variant_idx, variant_name in enumerate(['Zero-Shot', 'Adapted-Full', 'Adapted-LoRASA']):
            for n_agents in [2, 3, 4, 5]:
                rewards = [r['in_distribution']['LightMeta'][n_agents][variant_idx] for r in all_runs_results]
                writer.writerow([
                    'LightMeta', variant_name, n_agents,
                    f"{np.mean(rewards):.2f}",
                    f"{np.std(rewards):.2f}",
                    f"{np.percentile(rewards, 2.5):.2f}",
                    f"{np.percentile(rewards, 97.5):.2f}",
                    len(rewards)
                ])
        
        # Other methods
        for model in ['MAPPO', 'IPPO', 'MAML']:
            for n_agents in [2, 3, 4, 5]:
                idx = {'MAPPO': 1, 'IPPO': 0, 'MAML': 2}[model]
                rewards = [r['in_distribution'][model][n_agents][idx] for r in all_runs_results]
                writer.writerow([
                    model, 'Standard', n_agents,
                    f"{np.mean(rewards):.2f}",
                    f"{np.std(rewards):.2f}",
                    f"{np.percentile(rewards, 2.5):.2f}",
                    f"{np.percentile(rewards, 97.5):.2f}",
                    len(rewards)
                ])
    
    # Export scalability results
    with open('scalability_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'N_Agents', 'Reward_Mean', 'Reward_Std', 'Time_Mean', 'Time_Std', 
                        'Memory_Mean', 'Memory_Std', 'Latency_Mean', 'Latency_Std'])
        
        agent_counts = scalability_results[0]['agent_counts']
        for model in ['LightMeta', 'MAPPO', 'IPPO', 'MAML']:
            for i, n in enumerate(agent_counts):
                rewards = [r[model]['rewards'][i] for r in scalability_results]
                times = [r[model]['times'][i] for r in scalability_results]
                memory = [r[model]['memory'][i] for r in scalability_results]
                latency = [r[model]['latency'][i] for r in scalability_results]
                
                writer.writerow([
                    model, n,
                    f"{np.mean(rewards):.2f}", f"{np.std(rewards):.2f}",
                    f"{np.mean(times):.2f}", f"{np.std(times):.2f}",
                    f"{np.mean(memory):.2f}", f"{np.std(memory):.2f}",
                    f"{np.mean(latency):.3f}", f"{np.std(latency):.3f}"
                ])
    
    print("\n" + "="*60)
    print("✓ Results exported to:")
    print("  - main_comparison_results.csv")
    print("  - scalability_results.csv")
    print("="*60)


# --- Main Comparison ---
def run_comparison(args):
    seed, env_config, unseen_env_config = args
    print(f"\n--- Running Comparison for Seed: {seed} on {env_config['name']} ---")
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    env_fn, agent_obs_dim, num_actions = env_config['fn'], env_config['agent_obs_dim'], env_config['num_actions']
    input_dim = MAX_AGENTS * agent_obs_dim
    run_results = {
        "in_distribution": collections.defaultdict(dict), 
        "out_of_distribution": collections.defaultdict(dict),
        "compute_metrics": collections.defaultdict(dict),
        "learning_curves": collections.defaultdict(list)
    }
    
    print(f"[{seed}] Training LightMetaPolicy..."); light_meta_model = LightMetaPolicy(agent_obs_dim, num_actions); light_meta_trained, t, curve = train_light_meta_policy(light_meta_model, env_fn); run_results["compute_metrics"]["LightMeta"] = (sum(p.numel() for p in light_meta_model.parameters()), t); run_results["learning_curves"]["LightMeta"] = curve
    print(f"[{seed}] Training MAPPO..."); mappo_policy = MAPPOPolicy(agent_obs_dim, num_actions); mappo_trained, t, curve = train_mappo_policy(mappo_policy, env_fn); run_results["compute_metrics"]["MAPPO"] = (sum(p.numel() for p in mappo_policy.parameters()), t); run_results["learning_curves"]["MAPPO"] = curve
    print(f"[{seed}] Training IPPO..."); ippo_policies, t = train_ippo_policy(env_fn); run_results["compute_metrics"]["IPPO"] = (sum(p.numel() for p in list(ippo_policies.values())[0].policy.parameters()), t)
    print(f"[{seed}] Training MAML..."); maml_model = MAMLPolicy(input_dim, MAX_AGENTS * num_actions); maml_trained, t = meta_train_maml(maml_model, env_fn); run_results["compute_metrics"]["MAML"] = (sum(p.numel() for p in maml_model.parameters()), t)
    
    lorasa_model_for_counting = inject_lorasa(copy.deepcopy(light_meta_trained))
    run_results["compute_metrics"]["LightMeta_LoRASA"] = (sum(p.numel() for p in lorasa_model_for_counting.parameters() if p.requires_grad), 0)

    print(f"[{seed}] Evaluating models on in-distribution tasks...")
    for num_agents in [2, 3, 4, 5]:
        eval_env_fn = lambda seed=None: env_fn(num_agents=num_agents, seed=seed)
        
        lm_zero, _, lm_raw_zero = evaluate_policy(light_meta_trained, eval_env_fn)
        lm_full, _, lm_raw_full = evaluate_policy(fine_tune_standard(light_meta_trained, eval_env_fn), eval_env_fn)
        lm_lorasa, _, lm_raw_lorasa = evaluate_policy(fine_tune_with_lorasa(light_meta_trained, eval_env_fn), eval_env_fn)
        run_results["in_distribution"]["LightMeta"][num_agents] = (lm_zero, lm_full, lm_lorasa, lm_raw_lorasa) # Added raw for sig test

        mappo_reward, _, mappo_raw = evaluate_policy(mappo_trained, eval_env_fn, is_mappo=True); run_results["in_distribution"]["MAPPO"][num_agents] = (mappo_reward, mappo_raw)
        ippo_reward, _, ippo_raw = evaluate_policy(None, eval_env_fn, is_ippo=True, ippo_policies=ippo_policies); run_results["in_distribution"]["IPPO"][num_agents] = (ippo_reward, ippo_raw)
        maml_zero, _, maml_raw_zero = evaluate_policy(maml_trained, eval_env_fn); maml_adapted, _, maml_raw_adapted = evaluate_policy(fine_tune_maml_policy(maml_trained, eval_env_fn), eval_env_fn); run_results["in_distribution"]["MAML"][num_agents] = (maml_zero, maml_adapted, maml_raw_adapted)

    if not USE_SIMPLE_SPREAD:
        print(f"[{seed}] Evaluating models on out-of-distribution task...")
        unseen_env_fn = lambda seed=None: unseen_env_config['fn'](seed=seed)
        run_results["out_of_distribution"]["LightMeta"] = evaluate_policy(light_meta_trained, unseen_env_fn)
        run_results["out_of_distribution"]["MAPPO"] = evaluate_policy(mappo_trained, unseen_env_fn, is_mappo=True)
        run_results["out_of_distribution"]["IPPO"] = evaluate_policy(None, unseen_env_fn, is_ippo=True, ippo_policies=ippo_policies)
        run_results["out_of_distribution"]["MAML"] = evaluate_policy(maml_trained, unseen_env_fn)
    
    print(f"--- Finished Seed: {seed} ---")
    return run_results

def process_and_display_results(all_runs_results, env_name):
    print(f"\n\n{'='*20} AGGREGATED RESULTS ({env_name}) {'='*20}")
    agent_counts = [2, 3, 4, 5]
    model_names = ["LightMeta", "MAML", "MAPPO", "IPPO"]
    
    # --- In-Distribution Results Table ---
    print("\n" + "="*20 + " IN-DISTRIBUTION PERFORMANCE (TABLE) " + "="*20)
    header_parts = [f"N={n} Agents".ljust(18) for n in agent_counts]
    header = f"{'Model':<28} | " + " | ".join(header_parts)
    print(header); print("-" * len(header))
    for i, method in enumerate(["Zero-Shot", "Adapted (Full)", "Adapted (LoRASA)"]):
        label = f"LightMeta ({method})"
        row_data = [label.ljust(28)]
        for n in agent_counts:
            rewards = [r["in_distribution"]["LightMeta"][n][i] for r in all_runs_results]
            row_data.append(f"{np.mean(rewards):<8.2f} ± {np.std(rewards):<7.2f}")
        print(" | ".join(row_data))
    for i, method in enumerate(["Zero-Shot", "Adapted"]):
        label = f"MAML ({method})"
        row_data = [label.ljust(28)]
        for n in agent_counts:
            rewards = [r["in_distribution"]["MAML"][n][i] for r in all_runs_results]
            row_data.append(f"{np.mean(rewards):<8.2f} ± {np.std(rewards):<7.2f}")
        print(" | ".join(row_data))
    for model in ["MAPPO", "IPPO"]:
        row_data = [f"{model:<28}"]
        for n in agent_counts:
            rewards = [r["in_distribution"][model][n][0] for r in all_runs_results]
            row_data.append(f"{np.mean(rewards):<8.2f} ± {np.std(rewards):<7.2f}")
        print(" | ".join(row_data))
    
    # --- In-Distribution Results Plot ---
    plt.figure(figsize=(14, 8))

    # Better color scheme
    colors = {
        'LightMeta_zero': '#1f77b4',
        'LightMeta_full': '#ff7f0e', 
        'LightMeta_lorasa': '#2ca02c',
        'MAML_zero': '#d62728',
        'MAML_adapted': '#9467bd',
        'MAPPO': '#8c564b',
        'IPPO': '#e377c2'
    }

    # Compute means and confidence intervals
    lm_zero_means = [np.mean([r['in_distribution']['LightMeta'][n][0] for r in all_runs_results]) for n in agent_counts]
    lm_zero_ci_lower = [np.percentile([r['in_distribution']['LightMeta'][n][0] for r in all_runs_results], 2.5) for n in agent_counts]
    lm_zero_ci_upper = [np.percentile([r['in_distribution']['LightMeta'][n][0] for r in all_runs_results], 97.5) for n in agent_counts]

    lm_full_means = [np.mean([r['in_distribution']['LightMeta'][n][1] for r in all_runs_results]) for n in agent_counts]
    lm_full_ci_lower = [np.percentile([r['in_distribution']['LightMeta'][n][1] for r in all_runs_results], 2.5) for n in agent_counts]
    lm_full_ci_upper = [np.percentile([r['in_distribution']['LightMeta'][n][1] for r in all_runs_results], 97.5) for n in agent_counts]

    lm_lorasa_means = [np.mean([r['in_distribution']['LightMeta'][n][2] for r in all_runs_results]) for n in agent_counts]
    lm_lorasa_ci_lower = [np.percentile([r['in_distribution']['LightMeta'][n][2] for r in all_runs_results], 2.5) for n in agent_counts]
    lm_lorasa_ci_upper = [np.percentile([r['in_distribution']['LightMeta'][n][2] for r in all_runs_results], 97.5) for n in agent_counts]

    # Plot LightMeta variants with error bars
    plt.errorbar(agent_counts, lm_zero_means, 
                yerr=[np.array(lm_zero_means) - np.array(lm_zero_ci_lower),
                    np.array(lm_zero_ci_upper) - np.array(lm_zero_means)],
                label='LightMeta (Zero-Shot)', marker='o', linestyle='--', 
                linewidth=2.5, markersize=8, capsize=5, capthick=2,
                color=colors['LightMeta_zero'])

    plt.errorbar(agent_counts, lm_full_means,
                yerr=[np.array(lm_full_means) - np.array(lm_full_ci_lower),
                    np.array(lm_full_ci_upper) - np.array(lm_full_means)],
                label='LightMeta (Adapted-Full)', marker='s', linestyle='-.',
                linewidth=2.5, markersize=8, capsize=5, capthick=2,
                color=colors['LightMeta_full'])

    plt.errorbar(agent_counts, lm_lorasa_means,
                yerr=[np.array(lm_lorasa_means) - np.array(lm_lorasa_ci_lower),
                    np.array(lm_lorasa_ci_upper) - np.array(lm_lorasa_means)],
                label='LightMeta (Adapted-LoRASA)', marker='D', linestyle='-',
                linewidth=3, markersize=9, capsize=5, capthick=2,
                color=colors['LightMeta_lorasa'])

    # MAML
    maml_zero_means = [np.mean([r['in_distribution']['MAML'][n][0] for r in all_runs_results]) for n in agent_counts]
    maml_zero_ci_lower = [np.percentile([r['in_distribution']['MAML'][n][0] for r in all_runs_results], 2.5) for n in agent_counts]
    maml_zero_ci_upper = [np.percentile([r['in_distribution']['MAML'][n][0] for r in all_runs_results], 97.5) for n in agent_counts]

    maml_adapted_means = [np.mean([r['in_distribution']['MAML'][n][1] for r in all_runs_results]) for n in agent_counts]
    maml_adapted_ci_lower = [np.percentile([r['in_distribution']['MAML'][n][1] for r in all_runs_results], 2.5) for n in agent_counts]
    maml_adapted_ci_upper = [np.percentile([r['in_distribution']['MAML'][n][1] for r in all_runs_results], 97.5) for n in agent_counts]

    plt.errorbar(agent_counts, maml_zero_means,
                yerr=[np.array(maml_zero_means) - np.array(maml_zero_ci_lower),
                    np.array(maml_zero_ci_upper) - np.array(maml_zero_means)],
                label='MAML (Zero-Shot)', marker='^', linestyle='--',
                linewidth=2.5, markersize=8, capsize=5, capthick=2,
                color=colors['MAML_zero'])

    plt.errorbar(agent_counts, maml_adapted_means,
                yerr=[np.array(maml_adapted_means) - np.array(maml_adapted_ci_lower),
                    np.array(maml_adapted_ci_upper) - np.array(maml_adapted_means)],
                label='MAML (Adapted)', marker='v', linestyle='-',
                linewidth=2.5, markersize=8, capsize=5, capthick=2,
                color=colors['MAML_adapted'])

    # MAPPO and IPPO
    mappo_means = [np.mean([r['in_distribution']['MAPPO'][n][0] for r in all_runs_results]) for n in agent_counts]
    mappo_ci_lower = [np.percentile([r['in_distribution']['MAPPO'][n][0] for r in all_runs_results], 2.5) for n in agent_counts]
    mappo_ci_upper = [np.percentile([r['in_distribution']['MAPPO'][n][0] for r in all_runs_results], 97.5) for n in agent_counts]

    ippo_means = [np.mean([r['in_distribution']['IPPO'][n][0] for r in all_runs_results]) for n in agent_counts]
    ippo_ci_lower = [np.percentile([r['in_distribution']['IPPO'][n][0] for r in all_runs_results], 2.5) for n in agent_counts]
    ippo_ci_upper = [np.percentile([r['in_distribution']['IPPO'][n][0] for r in all_runs_results], 97.5) for n in agent_counts]

    plt.errorbar(agent_counts, mappo_means,
                yerr=[np.array(mappo_means) - np.array(mappo_ci_lower),
                    np.array(mappo_ci_upper) - np.array(mappo_means)],
                label='MAPPO', marker='*', linestyle='-',
                linewidth=2.5, markersize=11, capsize=5, capthick=2,
                color=colors['MAPPO'])

    plt.errorbar(agent_counts, ippo_means,
                yerr=[np.array(ippo_means) - np.array(ippo_ci_lower),
                    np.array(ippo_ci_upper) - np.array(ippo_means)],
                label='IPPO', marker='P', linestyle=':',
                linewidth=2.5, markersize=8, capsize=5, capthick=2,
                color=colors['IPPO'])

    plt.title(f'In-Distribution Performance on {env_name}\n(Aggregated over {len(SEEDS)} seeds with 95% CI)', 
            fontsize=16, fontweight='bold')
    plt.xlabel('Number of Agents', fontsize=14, fontweight='bold')
    plt.ylabel('Average Total Reward', fontsize=14, fontweight='bold')
    plt.xticks(agent_counts, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('in_distribution_performance.png', dpi=300, bbox_inches='tight')
    print("✓ In-distribution plot saved as 'in_distribution_performance.png'")
    plt.show()


    # --- Sample Efficiency Plot ---
    plt.figure(figsize=(14, 8))

    for model in ['LightMeta', 'MAPPO']:
        if model in all_runs_results[0]['learning_curves']:
            curves = [r['learning_curves'][model] for r in all_runs_results]
            all_steps = sorted(list(set([step for curve in curves for step, _ in curve])))
            
            aligned_curves = []
            for curve in curves:
                if not curve:
                    continue
                steps, rewards = zip(*curve)
                aligned_rewards = np.interp(all_steps, steps, rewards)
                aligned_curves.append(aligned_rewards)
            
            if aligned_curves:
                mean_curve = np.mean(aligned_curves, axis=0)
                std_curve = np.std(aligned_curves, axis=0)
                ci_lower = np.percentile(aligned_curves, 2.5, axis=0)
                ci_upper = np.percentile(aligned_curves, 97.5, axis=0)
                
                color = '#2ca02c' if model == 'LightMeta' else '#8c564b'
                
                plt.plot(all_steps, mean_curve, label=model, linewidth=3, color=color)
                plt.fill_between(all_steps, ci_lower, ci_upper, alpha=0.2, color=color)

    plt.title(f'Sample Efficiency on {env_name}\n(Training Reward vs Environment Steps, 95% CI)', 
            fontsize=16, fontweight='bold')
    plt.xlabel('Environment Interactions (Steps)', fontsize=14, fontweight='bold')
    plt.ylabel('Evaluation Return', fontsize=14, fontweight='bold')
    plt.legend(fontsize=13, loc='lower right', framealpha=0.95, edgecolor='black')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('sample_efficiency.png', dpi=300, bbox_inches='tight')
    print("✓ Sample efficiency plot saved as 'sample_efficiency.png'")
    plt.show()

    # --- NEW: Parameter Efficiency Visualization ---
    print("\nGenerating parameter efficiency plot...")
    plt.figure(figsize=(12, 7))

    try:
        # Extract parameter counts safely
        def safe_extract_params(metric_value):
            """Safely extract parameter count from various formats"""
            if isinstance(metric_value, (int, float)):
                return metric_value
            elif isinstance(metric_value, (tuple, list)) and len(metric_value) > 0:
                return metric_value[0]
            elif isinstance(metric_value, dict) and 'params' in metric_value:
                return metric_value['params']
            else:
                return None
        
        # Get first run results to extract values
        first_run = all_runs_results[0]['compute_metrics']
        
        param_dict = {
            'LightMeta\n(Zero-Shot)': safe_extract_params(first_run.get('LightMeta')),
            'LightMeta\n(LoRASA)': safe_extract_params(first_run.get('LightMetaLoRASA')),
            'MAPPO': safe_extract_params(first_run.get('MAPPO')),
            'IPPO': safe_extract_params(first_run.get('IPPO')),
            'MAML': safe_extract_params(first_run.get('MAML'))
        }
        
        # Filter out None values
        param_dict = {k: v for k, v in param_dict.items() if v is not None}
        
        if len(param_dict) >= 3:  # Need at least 3 models to plot
            models = list(param_dict.keys())
            params = list(param_dict.values())
            
            colors_bar = ['#1f77b4', '#2ca02c', '#8c564b', '#e377c2', '#d62728'][:len(models)]
            
            bars = plt.bar(models, params, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
            
            # Add value labels on bars
            for i, (bar, param) in enumerate(zip(bars, params)):
                height = bar.get_height()
                label = f'{param/1000:.1f}K' if param < 1000000 else f'{param/1000000:.2f}M'
                plt.text(bar.get_x() + bar.get_width()/2., height + max(params)*0.02,
                        label, ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                # Add trainable params annotation for LoRASA
                if 'LoRASA' in models[i]:
                    plt.text(bar.get_x() + bar.get_width()/2., height/2,
                            f'Only {param/1000:.1f}K\ntrainable',
                            ha='center', va='center', fontsize=10, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.title('Total Model Parameters: Comparison Across Methods', fontsize=16, fontweight='bold')
            plt.ylabel('Number of Parameters', fontsize=14, fontweight='bold')
            plt.xlabel('Method', fontsize=14, fontweight='bold')
            plt.yscale('log')
            plt.grid(True, axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig('parameter_efficiency.png', dpi=300, bbox_inches='tight')
            print("✓ Parameter efficiency plot saved as 'parameter_efficiency.png'")
            plt.show()
        else:
            print(f"⚠️  Insufficient data for parameter efficiency plot (only {len(param_dict)} models)")
            
    except Exception as e:
        print(f"⚠️  Skipping parameter efficiency plot due to error: {e}")
        import traceback
        traceback.print_exc()




    # --- OOD, Adaptation Gain, and Compute Tables ---
    if not USE_SIMPLE_SPREAD:
        print("\n" + "="*20 + " OUT-OF-DISTRIBUTION (OOD) RESULTS " + "="*20)
        print(f"{'Model':<20} | {'Mean Reward':<15} | {'Std Dev':<15}")
        print("-"*55)
        for model in model_names:
            if "out_of_distribution" in all_runs_results[0] and model in all_runs_results[0]["out_of_distribution"]:
                rewards = [r["out_of_distribution"][model][0] for r in all_runs_results]
                print(f"{model:<20} | {np.mean(rewards):<15.2f} | {np.std(rewards):<15.2f}")
        
        print("\n" + "="*20 + " ADAPTATION GAIN (N=5 AGENTS) " + "="*20)
        print(f"{'Model':<25} | {'Gain':<15}")
        print("-"*45)
        lm_zero = [r["in_distribution"]["LightMeta"][5][0] for r in all_runs_results]
        lm_lorasa = [r["in_distribution"]["LightMeta"][5][2] for r in all_runs_results]
        maml_zero = [r["in_distribution"]["MAML"][5][0] for r in all_runs_results]
        maml_adapted = [r["in_distribution"]["MAML"][5][1] for r in all_runs_results]
        print(f"{'LightMeta (LoRASA)':<25} | {np.mean(lm_lorasa) - np.mean(lm_zero):<15.2f}")
        print(f"{'MAML':<25} | {np.mean(maml_adapted) - np.mean(maml_zero):<15.2f}")
        
    print("\n" + "="*70)
    print("DEBUG: Data Structure")
    print("="*70)
    if len(all_runs_results) > 0:
        first_run = all_runs_results[0]['in_distribution']
        for model in ['LightMeta', 'MAPPO', 'IPPO', 'MAML']:
            print(f"\n{model}:")
            if 5 in first_run[model]:
                data = first_run[model][5]
                print(f"  Type: {type(data)}")
                print(f"  Value: {data}")
                if isinstance(data, (list, tuple)):
                    print(f"  Length: {len(data)}")
                    for i, val in enumerate(data):
                        print(f"    [{i}]: {val} (type: {type(val)})")

    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTS (N=5 AGENTS)")
    print("="*70)

    from scipy.stats import ttest_ind

    try:
        # Extract rewards for N=5 agents correctly
        # Structure: all_runs_results[run_index]['in_distribution'][model][agent_count][variant_index]
        lm_raw = []
        mappo_raw = []
        ippo_raw = []
        maml_raw = []
        
        for r in all_runs_results:
            # LightMeta LoRASA adapted (variant index 2)
            if 5 in r['in_distribution']['LightMeta']:
                lm_val = r['in_distribution']['LightMeta'][5][2]  # Index 2 = LoRASA adapted
                if isinstance(lm_val, (int, float)):
                    lm_raw.append(lm_val)
                elif isinstance(lm_val, (list, tuple)) and len(lm_val) > 0:
                    lm_raw.append(lm_val[0] if isinstance(lm_val[0], (int, float)) else np.mean(lm_val))
            
            # MAPPO (variant index 1 or 0, depending on structure)
            if 5 in r['in_distribution']['MAPPO']:
                mappo_val = r['in_distribution']['MAPPO'][5]
                if isinstance(mappo_val, (list, tuple)):
                    # MAPPO stores as [mean_reward] or (mean_reward, std, rewards)
                    mappo_val = mappo_val[0] if len(mappo_val) > 0 else mappo_val
                if isinstance(mappo_val, (int, float)):
                    mappo_raw.append(mappo_val)
            
            # IPPO
            if 5 in r['in_distribution']['IPPO']:
                ippo_val = r['in_distribution']['IPPO'][5]
                if isinstance(ippo_val, (list, tuple)):
                    ippo_val = ippo_val[0] if len(ippo_val) > 0 else ippo_val
                if isinstance(ippo_val, (int, float)):
                    ippo_raw.append(ippo_val)
            
            # MAML adapted (variant index 1 or 2)
            if 5 in r['in_distribution']['MAML']:
                maml_vals = r['in_distribution']['MAML'][5]
                if isinstance(maml_vals, (list, tuple)) and len(maml_vals) > 1:
                    maml_val = maml_vals[1]  # Index 1 = adapted
                    if isinstance(maml_val, (int, float)):
                        maml_raw.append(maml_val)
        
        # Check if we have enough data
        if len(lm_raw) > 1 and len(mappo_raw) > 1:
            print(f"\nSample sizes: LightMeta N={len(lm_raw)}, MAPPO N={len(mappo_raw)}, IPPO N={len(ippo_raw)}, MAML N={len(maml_raw)}")
            
            # Ensure equal lengths for comparison (truncate to minimum)
            min_len = min(len(lm_raw), len(mappo_raw), len(ippo_raw), len(maml_raw))
            lm_raw = lm_raw[:min_len]
            mappo_raw = mappo_raw[:min_len]
            ippo_raw = ippo_raw[:min_len]
            maml_raw = maml_raw[:min_len]
            
            # LightMeta vs MAPPO
            if len(mappo_raw) > 1:
                t_stat, p_val = ttest_ind(lm_raw, mappo_raw)
                sig = "✓ SIGNIFICANT" if p_val < 0.05 else "✗ Not significant"
                print(f"  LightMeta vs MAPPO:  t={t_stat:6.3f}, p={p_val:.4f}  {sig}")
            
            # LightMeta vs IPPO
            if len(ippo_raw) > 1:
                t_stat, p_val = ttest_ind(lm_raw, ippo_raw)
                sig = "✓ SIGNIFICANT" if p_val < 0.05 else "✗ Not significant"
                print(f"  LightMeta vs IPPO:   t={t_stat:6.3f}, p={p_val:.4f}  {sig}")
            
            # LightMeta vs MAML
            if len(maml_raw) > 1:
                t_stat, p_val = ttest_ind(lm_raw, maml_raw)
                sig = "✓ SIGNIFICANT" if p_val < 0.05 else "✗ Not significant"
                print(f"  LightMeta vs MAML:   t={t_stat:6.3f}, p={p_val:.4f}  {sig}")
            
            # 95% Confidence Intervals
            print(f"\n95% Confidence Intervals:")
            print(f"  LightMeta: [{np.percentile(lm_raw, 2.5):7.2f}, {np.percentile(lm_raw, 97.5):7.2f}]")
            print(f"  MAPPO:     [{np.percentile(mappo_raw, 2.5):7.2f}, {np.percentile(mappo_raw, 97.5):7.2f}]")
            print(f"  IPPO:      [{np.percentile(ippo_raw, 2.5):7.2f}, {np.percentile(ippo_raw, 97.5):7.2f}]")
            print(f"  MAML:      [{np.percentile(maml_raw, 2.5):7.2f}, {np.percentile(maml_raw, 97.5):7.2f}]")
        
        elif len(lm_raw) == 1:
            print(f"\n⚠️  Only {len(lm_raw)} seed(s) - statistical tests require at least 2 seeds")
            print(f"  LightMeta mean: {np.mean(lm_raw):.2f}")
            print(f"  MAPPO mean:     {np.mean(mappo_raw):.2f}")
            print(f"  IPPO mean:      {np.mean(ippo_raw):.2f}")
            print(f"  MAML mean:      {np.mean(maml_raw):.2f}")
        else:
            print(f"\n⚠️  Insufficient data for statistical tests")
            print(f"  LightMeta: {len(lm_raw)} samples")
            print(f"  MAPPO: {len(mappo_raw)} samples")
            print(f"  IPPO: {len(ippo_raw)} samples")
            print(f"  MAML: {len(maml_raw)} samples")

    except Exception as e:
        print(f"\n⚠️  Error in statistical tests: {e}")
        import traceback
        traceback.print_exc()


    print("\n" + "="*20 + " COMPUTE & TRAINING TIME METRICS " + "="*20)
    print(f"{'Model':<25} | {'Parameters':<20} | {'Training Time (s)':<20}")
    print("-"*70)
    all_model_names = model_names + ["LightMeta_LoRASA"]
    for model in all_model_names:
        if model in all_runs_results[0]["compute_metrics"]:
            params = [r["compute_metrics"][model][0] for r in all_runs_results]
            times = [r["compute_metrics"][model][1] for r in all_runs_results]
            
            if model == "LightMeta_LoRASA":
                full_params = [r["compute_metrics"]["LightMeta"][0] for r in all_runs_results]
                param_str = f"{np.mean(params):.0f} ({np.mean(params)/np.mean(full_params):.2%})"
                time_str = "N/A (part of FT)"
                display_name = "LightMeta (LoRASA Adapters)"
            else:
                param_str = f"{np.mean(params):.0f}"
                time_str = f"{np.mean(times):.2f} ± {np.std(times):.2f}"
                display_name = model

            print(f"{display_name:<25} | {param_str:<20} | {time_str:<20}")
            

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("⚠️  RUNNING IN TEST MODE - QUICK VALIDATION ⚠️")
    print("="*60)
    print(f"Settings:")
    print(f"  - Seeds: {SEEDS}")
    print(f"  - Scalability agent counts: {SCALABILITY_AGENT_COUNTS}")
    print(f"  - Meta iterations: {META_ITERATIONS}")
    print(f"  - Scalability meta iterations: {SCALABILITY_META_ITERS}")
    print("="*60 + "\n")
    max_ss_obs_dim = SimpleSpreadWrapper.get_max_obs_dim(MAX_AGENTS)
    ss_action_dim = SimpleSpreadWrapper(num_agents=2).action_dim
    
    ENV_CONFIGS = {
        "SimpleSpread": {"name": "Simple Spread", "fn": SimpleSpreadWrapper, "agent_obs_dim": max_ss_obs_dim, "num_actions": ss_action_dim},
        "CustomEnv": {"name": "Custom Environment", "fn": CustomMultiAgentEnv, "agent_obs_dim": 4, "num_actions": 2}
    }
    
    UNSEEN_ENV_CONFIG = { "UnseenCustomEnv": {"name": "Unseen Custom", "fn": UnseenCustomMultiAgentEnv} }
    
    config = ENV_CONFIGS["SimpleSpread"] if USE_SIMPLE_SPREAD else ENV_CONFIGS["CustomEnv"]
    unseen_config = UNSEEN_ENV_CONFIG["UnseenCustomEnv"]
    
    # --- Run Scalability Tests First ---
    if SCALABILITY_TEST:
        print("\n" + "="*70)
        print("PHASE 1: SCALABILITY EXPERIMENTS")
        print("="*70)
        print(f"Testing agent counts: {SCALABILITY_AGENT_COUNTS}")
        print(f"Using {len(SEEDS[:3])} seeds for scalability: {SEEDS[:3]}")
        
        scalability_results = []
        for seed in SEEDS[:3]:  # Use 3 seeds for scalability
            result = run_scalability_test(seed, config)
            scalability_results.append(result)
        
        plot_scalability_results(scalability_results)
    else:
        scalability_results = None
    
    # --- Run Main Comparison ---
    print("\n" + "="*70)
    print("PHASE 2: MAIN COMPARISON EXPERIMENTS")
    print("="*70)
    print(f"Using {len(SEEDS)} seeds: {SEEDS}")
    
    with mp.Pool(processes=NUM_WORKERS) as pool:
        args_for_pool = [(seed, config, unseen_config) for seed in SEEDS]
        all_runs_results = pool.map(run_comparison, args_for_pool)
    
    process_and_display_results(all_runs_results, config["name"])
    
    # --- Export Results ---
    if scalability_results:
        export_results_to_csv(all_runs_results, scalability_results)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)

