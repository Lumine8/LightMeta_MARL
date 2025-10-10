import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import matplotlib.pyplot as plt
import time
import collections
import multiprocessing as mp
from pettingzoo.mpe import simple_spread_v3

# --- SCRIPT CONFIGURATION ---
SEEDS = [42, 123, 888, 2024, 7]
NUM_WORKERS = 6
MAX_AGENTS = 5
EPISODES = 20
META_ITERATIONS = 500

# --- LoRASA Layer ---
class LoRASALayer(nn.Module):
    def __init__(self, linear_layer, rank=4):
        super().__init__()
        self.linear = linear_layer
        in_features, out_features = linear_layer.in_features, linear_layer.out_features
        
        # Freeze the original linear layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_scale = nn.Parameter(torch.ones(out_features))
        self.lora_shift = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

    def forward(self, x):
        frozen_output = self.linear(x)
        adapter_output = (x @ self.lora_A @ self.lora_B) * self.lora_scale + self.lora_shift
        return frozen_output + adapter_output

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
        self.agent_positions = local_random_state.randint(20, 580, (self.num_agents, 2))
        self.goal_zones = local_random_state.randint(50, 550, (self.num_agents, 2))
        self.agent_states = np.random.uniform(low=-1, high=1, size=(self.num_agents, 4))
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
            self.agent_positions[move_mask] += np.random.randint(-self.move_range, self.move_range, size=(np.sum(move_mask), 2))
        for i in range(self.num_agents):
            rewards[i] += max(0, 100 - np.linalg.norm(self.agent_positions[i] - self.goal_zones[i])) / 100
        self.steps += 1; terminated = self.steps >= self.max_steps
        obs = self.agent_states.flatten()
        padded_obs = np.zeros((MAX_AGENTS * 4,), dtype=np.float32); padded_obs[:len(obs)] = obs
        return padded_obs, np.sum(rewards) / self.num_agents, terminated, False, {}

class UnseenCustomMultiAgentEnv(CustomMultiAgentEnv):
    def randomize_environment(self):
        super().randomize_environment()
        local_random_state = np.random.RandomState(self.seed)
        self.move_range = local_random_state.randint(25, 50)
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
        self.num_agents = num_agents; self.seed = seed
        self.max_obs_dim_per_agent = self.get_max_obs_dim(MAX_AGENTS)
        self.action_dim = self.env.action_space("agent_0").n
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_AGENTS * self.max_obs_dim_per_agent,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.action_dim] * MAX_AGENTS)
        self.agents = [f'agent_{i}' for i in range(self.num_agents)]
        self.reset(seed=self.seed)
    def reset(self, seed=None, options=None):
        obs_dict, infos = self.env.reset(seed=seed if seed is not None else self.seed)
        return self._flatten_obs(obs_dict), infos
    def step(self, actions):
        action_dict = {agent: actions[i] for i, agent in enumerate(self.agents)}
        next_obs_dict, rewards_dict, terminateds_dict, _, _ = self.env.step(action_dict)
        reward = sum(rewards_dict.values()) / self.num_agents if self.num_agents > 0 else 0
        terminated = all(terminateds_dict.values())
        return self._flatten_obs(next_obs_dict), reward, terminated, False, {}
    def _flatten_obs(self, obs_dict):
        padded_obs = np.zeros((MAX_AGENTS * self.max_obs_dim_per_agent,), dtype=np.float32)
        for i, agent_name in enumerate(self.agents):
            if agent_name in obs_dict:
                obs = obs_dict[agent_name]
                padded_obs[i*self.max_obs_dim_per_agent : i*self.max_obs_dim_per_agent + len(obs)] = obs
        return padded_obs

# --- Model ---
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


def inject_lorasa(model, rank=4):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRASALayer(module, rank=rank))
        elif len(list(module.children())) > 0: # Recurse for nn.Sequential
            inject_lorasa(module, rank=rank)
    return model

def freeze_base_parameters(model):
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

# --- Training & Fine-Tuning Functions ---
def train_model(model, env_fn, iterations=META_ITERATIONS, inner_rollouts=40, gamma=0.99, entropy_coef=0.01, lr=0.0003):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(iterations):
        num_agents = np.random.choice([2, 3, 4, 5])
        env = env_fn(num_agents=num_agents)
        obs, _ = env.reset()
        log_probs, rewards, values, entropies = [], [], [], []
        for _ in range(inner_rollouts):
            action_logits, value = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            dist = torch.distributions.Categorical(logits=action_logits[:, :num_agents, :])
            actions = dist.sample()
            next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
            log_probs.append(dist.log_prob(actions).mean()); rewards.append(reward); values.append(value); entropies.append(dist.entropy().mean())
            obs = next_obs
            if terminated: obs, _ = env.reset()
        returns = []
        discounted_reward = 0
        for r in reversed(rewards): discounted_reward = r + gamma * discounted_reward; returns.insert(0, discounted_reward)
        returns, values = torch.tensor(returns, dtype=torch.float32), torch.cat(values).squeeze()
        advantages = returns - values.detach()
        policy_loss = -(torch.stack(log_probs) * advantages).mean()
        value_loss = nn.MSELoss()(values, returns)
        loss = policy_loss + 0.5 * value_loss - entropy_coef * torch.stack(entropies).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model

def fine_tune_standard(model, env_fn, max_steps=40):
    tuned_model = copy.deepcopy(model)
    optimizer = optim.Adam(tuned_model.parameters(), lr=0.0001)
    env = env_fn(num_agents=5)
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
    env = env_fn(num_agents=5)
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

def evaluate_policy(model, env_fn, episodes=EPISODES):
    rewards = []
    for ep in range(episodes):
        env = env_fn(num_agents=5, seed=1000 + ep)
        obs, _ = env.reset()
        total_reward = 0; done = False
        while not done:
            with torch.no_grad():
                action_logits, _ = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                actions = torch.argmax(action_logits[:, :env.num_agents, :], dim=-1).numpy().flatten()
            obs, reward, terminated, _, _ = env.step(actions)
            total_reward += reward; done = terminated
        rewards.append(total_reward)
    return np.mean(rewards)

# --- Worker Function ---
def run_single_seed(args):
    seed, all_env_configs = args
    print(f"  Starting seed {seed}...")
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    
    seed_results = {}

    for domain in ["CustomEnv", "SimpleSpread"]:
        train_env_config = all_env_configs[domain]
        train_env_fn = train_env_config["fn"]
        
        start_time = time.time()
        base_model = LightMetaPolicy(train_env_config["agent_obs_dim"], train_env_config["num_actions"])
        trained_model = train_model(base_model, train_env_fn)
        train_time = time.time() - start_time
        
        seed_results[f"{domain} Train Time"] = train_time
        seed_results[f"{domain} Full Params"] = sum(p.numel() for p in base_model.parameters())
        
        lorasa_model_for_counting = inject_lorasa(copy.deepcopy(base_model))
        seed_results[f"{domain} LoRASA Params"] = sum(p.numel() for p in lorasa_model_for_counting.parameters() if p.requires_grad)

        if domain == "CustomEnv":
            eval_configs = {"Custom ID": all_env_configs["CustomEnv"], "Unseen OOD": all_env_configs["UnseenEnv"]}
        else:
            eval_configs = {"Simple Spread": all_env_configs["SimpleSpread"]}

        for eval_name, eval_config in eval_configs.items():
            eval_env_fn = eval_config["fn"]
            
            seed_results[f"{eval_name} Zero-Shot"] = evaluate_policy(trained_model, eval_env_fn)
            
            adapted_full_model = fine_tune_standard(trained_model, eval_env_fn)
            seed_results[f"{eval_name} Adapted (Full)"] = evaluate_policy(adapted_full_model, eval_env_fn)
            
            adapted_lorasa_model = fine_tune_with_lorasa(trained_model, eval_env_fn)
            seed_results[f"{eval_name} Adapted (LoRASA)"] = evaluate_policy(adapted_lorasa_model, eval_env_fn)

    print(f"  Finished seed {seed}.")
    return seed_results

# --- Main Orchestrator ---
if __name__ == "__main__":
    max_ss_obs_dim = SimpleSpreadWrapper.get_max_obs_dim(MAX_AGENTS)
    ss_action_dim = SimpleSpreadWrapper(num_agents=2).action_dim
    
    ALL_ENV_CONFIGS = {
        "CustomEnv": {"fn": CustomMultiAgentEnv, "agent_obs_dim": 4, "num_actions": 2},
        "UnseenEnv": {"fn": UnseenCustomMultiAgentEnv},
        "SimpleSpread": {"fn": SimpleSpreadWrapper, "agent_obs_dim": max_ss_obs_dim, "num_actions": ss_action_dim}
    }

    args_for_pool = [(seed, ALL_ENV_CONFIGS) for seed in SEEDS]
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results_from_seeds = pool.map(run_single_seed, args_for_pool)
    
    # --- Aggregate and Print Final Summary ---
    eval_names = ["Custom ID", "Unseen OOD", "Simple Spread"]
    adaptation_methods = ["Zero-Shot", "Adapted (Full)", "Adapted (LoRASA)"]
    
    aggregated_results = collections.defaultdict(list)
    for seed_result in results_from_seeds:
        for key, value in seed_result.items():
            aggregated_results[key].append(value)
            
    print("\n\n" + "="*30 + " ADAPTATION METHOD COMPARISON " + "="*30)
    header = f"{'Environment':<25} | " + " | ".join([f'{method:<20}' for method in adaptation_methods])
    print(header)
    print("-" * (len(header) + 2))

    for env_name in eval_names:
        row_data = [f"{env_name:<25}"]
        for method in adaptation_methods:
            key = f"{env_name} {method}"
            mean_reward = np.mean(aggregated_results[key])
            std_reward = np.std(aggregated_results[key])
            row_data.append(f"{mean_reward:<8.2f} ± {std_reward:<7.2f}")
        print(" | ".join(row_data))

    print("\n\n" + "="*30 + " COMPUTATIONAL COST COMPARISON " + "="*30)
    header = f"{'Domain':<25} | {'Total Train Time (s)':<25} | {'Full Model Params':<20} | {'LoRASA Params':<20}"
    print(header)
    print("-" * (len(header) + 2))
    for domain in ["CustomEnv", "SimpleSpread"]:
        train_time_mean = np.mean(aggregated_results[f"{domain} Train Time"])
        train_time_std = np.std(aggregated_results[f"{domain} Train Time"])
        full_params = np.mean(aggregated_results[f"{domain} Full Params"])
        lorasa_params = np.mean(aggregated_results[f"{domain} LoRASA Params"])
        row_data = [
            f"{domain:<25}",
            f"{train_time_mean:<10.2f} ± {train_time_std:<7.2f}",
            f"{full_params:<20.0f}",
            f"{lorasa_params:<20.0f} ({lorasa_params/full_params:.2%})"
        ]
        print(" | ".join(row_data))
