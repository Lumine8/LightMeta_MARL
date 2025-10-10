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
import multiprocessing as mp
from pettingzoo.mpe import simple_spread_v3

# ============================================================================
# CONFIGURATION
# ============================================================================

SEEDS = [42, 123, 888, 2024, 7]  # 5 seeds
MAX_AGENTS = 5
META_ITERATIONS = 800
EPISODES = 20
NUM_WORKERS = 5  # Run all 5 seeds in parallel

# Test both environments
TEST_SIMPLE_SPREAD = True
TEST_CUSTOM_ENV = True

# ============================================================================
# ABLATION CONFIGS
# ============================================================================

TUNING_CONFIGS = {
    "baseline": {
        "lr": 0.0001,
        "entropy_coef": 0.01,
        "use_layer_norm": False,
        "use_relational_bias": False,
        "use_residual": False,
        "description": "No LayerNorm, no residual, no bias"
    },
    
    "with_layernorm": {
        "lr": 0.0001,
        "entropy_coef": 0.01,
        "use_layer_norm": True,
        "use_relational_bias": False,
        "use_residual": False,
        "description": "With LayerNorm only"
    },
    
    "with_residual": {
        "lr": 0.0001,
        "entropy_coef": 0.01,
        "use_layer_norm": False,
        "use_relational_bias": False,
        "use_residual": True,
        "description": "With residual connection only"
    },
    
    "with_relational_bias": {
        "lr": 0.0001,
        "entropy_coef": 0.01,
        "use_layer_norm": False,
        "use_relational_bias": True,
        "use_residual": False,
        "description": "With relational bias only"
    },
    
    "layernorm_and_residual": {
        "lr": 0.0001,
        "entropy_coef": 0.01,
        "use_layer_norm": True,
        "use_relational_bias": False,
        "use_residual": True,
        "description": "LayerNorm + residual"
    },
    
    "champion_final": {
        "lr": 0.0001,
        "entropy_coef": 0.01,
        "use_layer_norm": False,
        "use_relational_bias": False,
        "use_residual": True,
        "description": "Final champion (residual only)"
    },
    
    "all_features": {
        "lr": 0.0001,
        "entropy_coef": 0.01,
        "use_layer_norm": True,
        "use_relational_bias": True,
        "use_residual": True,
        "description": "All features enabled"
    },
}

# ============================================================================
# SIMPLE SPREAD ENVIRONMENT
# ============================================================================

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
        self.agents = [f'agent_{i}' for i in range(self.num_agents)]
        self.reset(seed=self.seed)
    
    def reset(self, seed=None, options=None):
        obs_dict, infos = self.env.reset(seed=seed if seed is not None else self.seed)
        return self._flatten_obs(obs_dict), infos
    
    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        
        actions = np.array(actions).flatten()[:self.num_agents]
        action_dict = {self.agents[i]: int(actions[i]) for i in range(self.num_agents)}
        
        next_obs_dict, rewards_dict, terminateds_dict, _, _ = self.env.step(action_dict)
        reward = sum(rewards_dict.values()) / self.num_agents if self.num_agents > 0 else 0
        terminated = all(terminateds_dict.values())
        return self._flatten_obs(next_obs_dict), reward, terminated, False, {}
    
    def _flatten_obs(self, obs_dict):
        padded_obs = np.zeros((MAX_AGENTS, self.max_obs_dim_per_agent), dtype=np.float32)
        for i, agent_name in enumerate(self.agents):
            if agent_name in obs_dict:
                obs = obs_dict[agent_name]
                padded_obs[i, :len(obs)] = obs
        return padded_obs.flatten()

# ============================================================================
# CUSTOM ENVIRONMENT
# ============================================================================

class CustomMultiAgentEnv(gym.Env):
    def __init__(self, num_agents=2, seed=None):
        super().__init__()
        self.num_agents = num_agents
        self.observation_space = spaces.Box(low=-1, high=1, shape=(MAX_AGENTS * 4,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2] * MAX_AGENTS)
        self.max_steps = 200
        self.steps = 0
        self.seed = seed
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
        self.randomize_environment()
        self.steps = 0
        obs = self.agent_states.flatten()
        padded_obs = np.zeros((MAX_AGENTS * 4,), dtype=np.float32)
        padded_obs[:len(obs)] = obs
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
        self.steps += 1
        terminated = self.steps >= self.max_steps
        
        noisy_obs = self.agent_states + np.random.normal(0, 0.2, size=self.agent_states.shape)
        obs = noisy_obs.flatten()
        padded_obs = np.zeros((MAX_AGENTS * 4,), dtype=np.float32)
        padded_obs[:len(obs)] = obs
        return padded_obs, np.sum(rewards) / self.num_agents, terminated, False, {}

# ============================================================================
# LIGHTMETA POLICY
# ============================================================================

class LightMetaPolicy(nn.Module):
    def __init__(self, agent_obs_dim, num_actions, use_layer_norm=False, use_residual=True, use_relational_bias=False):
        super().__init__()
        self.agent_dim = agent_obs_dim
        self.num_actions = num_actions
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.use_relational_bias = use_relational_bias
        self.d_model = 64
        
        self.input_proj = nn.Linear(self.agent_dim, self.d_model)
        self.key_transform = nn.Linear(self.d_model, self.d_model)
        self.query_transform = nn.Linear(self.d_model, self.d_model)
        self.value_transform = nn.Linear(self.d_model, self.d_model)
        
        if self.use_relational_bias:
            self.agent_relation = nn.Linear(self.d_model, self.d_model)
        
        if self.use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(self.d_model)
            self.layer_norm2 = nn.LayerNorm(self.d_model)
        
        self.fc_out = nn.Linear(self.d_model, self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
        self.action_head = nn.Linear(self.d_model, MAX_AGENTS * self.num_actions)
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        agents = x.view(batch_size, MAX_AGENTS, self.agent_dim)
        
        projected_agents = self.input_proj(agents)
        queries = self.query_transform(projected_agents)
        keys = self.key_transform(projected_agents)
        values = self.value_transform(projected_agents)
        
        attention = torch.matmul(queries, keys.transpose(-2, -1)) / (self.d_model ** 0.5)
        
        if self.use_relational_bias:
            rel_bias_logits = self.agent_relation(queries)
            attention += torch.matmul(rel_bias_logits, queries.transpose(-2, -1)) / (self.d_model ** 0.5)
        
        attention = torch.softmax(attention, dim=-1)
        context = torch.matmul(attention, values)
        context = self.fc_out(context)
        
        if self.use_residual:
            context = context + projected_agents
        
        if self.use_layer_norm:
            context = self.layer_norm1(context)
        
        agent_mask = (agents.abs().sum(dim=-1, keepdim=True) > 0.01).float()
        pooled = (context * agent_mask).sum(dim=1) / (agent_mask.sum(dim=1) + 1e-8)
        
        ffn_out = self.ffn(pooled)
        
        if self.use_residual:
            ffn_out = ffn_out + pooled
        
        if self.use_layer_norm:
            ffn_out = self.layer_norm2(ffn_out)
        
        action_logits = self.action_head(ffn_out).view(batch_size, MAX_AGENTS, self.num_actions)
        value = self.value_head(pooled)
        return action_logits, value

# ============================================================================
# TRAINING & EVALUATION FUNCTIONS
# ============================================================================

def train_light_meta(model, env_fn, config, meta_iterations=800):
    """Meta-train with variance reduction fixes"""
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    for iteration in range(meta_iterations):
        num_agents = np.random.choice([2, 3, 4, 5])
        env = env_fn(num_agents=num_agents)
        obs, _ = env.reset()
        
        log_probs, rewards, values = [], [], []
        for _ in range(128):
            with torch.no_grad():
                action_logits, value = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                dist = torch.distributions.Categorical(logits=action_logits[:, :num_agents, :])
                actions = dist.sample()
            
            next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
            
            action_logits, value = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            dist = torch.distributions.Categorical(logits=action_logits[:, :num_agents, :])
            log_probs.append(dist.log_prob(actions).mean())
            rewards.append(reward)
            values.append(value)
            
            obs = next_obs
            if terminated:
                obs, _ = env.reset()
        
        if len(rewards) == 0:
            continue
        
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + 0.99 * discounted_reward
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        values = torch.cat(values).squeeze()
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        policy_loss = -(torch.stack(log_probs) * advantages).mean()
        value_loss = nn.MSELoss()(values, returns)
        
        loss = policy_loss + 0.5 * value_loss - config['entropy_coef'] * dist.entropy().mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
    
    return model

def evaluate(model, env_fn, episodes=20):
    """Evaluate model"""
    total_rewards = []
    
    for ep in range(episodes):
        env = env_fn(num_agents=5, seed=1000 + ep)
        obs, _ = env.reset()
        episode_reward = 0
        
        for _ in range(200):
            with torch.no_grad():
                action_logits, _ = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                actions = torch.argmax(action_logits[:, :5, :], dim=-1).numpy().flatten()
            
            obs, reward, terminated, _, _ = env.step(actions)
            episode_reward += reward
            if terminated:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards), np.std(total_rewards)

# ============================================================================
# WORKER FUNCTION FOR MULTIPROCESSING
# ============================================================================

def run_single_seed(args):
    """Worker function to train and evaluate a single seed"""
    config_name, config, seed = args
    
    print(f"[Seed {seed}] Starting {config_name}...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    results_seed = {'seed': seed}
    
    # ===== SIMPLE SPREAD =====
    if TEST_SIMPLE_SPREAD:
        max_obs_dim_ss = SimpleSpreadWrapper.get_max_obs_dim(MAX_AGENTS)
        action_dim_ss = SimpleSpreadWrapper(num_agents=2).action_dim
        env_fn_ss = lambda num_agents=None, seed=None: SimpleSpreadWrapper(num_agents=num_agents, seed=seed)
        
        model_ss = LightMetaPolicy(
            max_obs_dim_ss, action_dim_ss,
            use_layer_norm=config['use_layer_norm'],
            use_residual=config['use_residual'],
            use_relational_bias=config['use_relational_bias']
        )
        
        start_time = time.time()
        model_ss = train_light_meta(model_ss, env_fn_ss, config, meta_iterations=META_ITERATIONS)
        train_time_ss = time.time() - start_time
        
        reward_ss, std_ss = evaluate(model_ss, env_fn_ss, episodes=EPISODES)
        
        results_seed['simple_spread'] = {
            'reward': reward_ss,
            'std': std_ss,
            'train_time': train_time_ss
        }
        
        print(f"[Seed {seed}] Simple Spread: {reward_ss:.2f} ± {std_ss:.2f}, Time: {train_time_ss:.1f}s")
    
    # ===== CUSTOM ENVIRONMENT =====
    if TEST_CUSTOM_ENV:
        env_fn_custom = lambda num_agents=None, seed=None: CustomMultiAgentEnv(num_agents=num_agents if num_agents else 5, seed=seed)
        
        model_custom = LightMetaPolicy(
            4, 2,
            use_layer_norm=config['use_layer_norm'],
            use_residual=config['use_residual'],
            use_relational_bias=config['use_relational_bias']
        )
        
        start_time = time.time()
        model_custom = train_light_meta(model_custom, env_fn_custom, config, meta_iterations=META_ITERATIONS)
        train_time_custom = time.time() - start_time
        
        reward_custom, std_custom = evaluate(model_custom, env_fn_custom, episodes=EPISODES)
        
        results_seed['custom_env'] = {
            'reward': reward_custom,
            'std': std_custom,
            'train_time': train_time_custom
        }
        
        print(f"[Seed {seed}] Custom Env: {reward_custom:.2f} ± {std_custom:.2f}, Time: {train_time_custom:.1f}s")
    
    print(f"[Seed {seed}] Completed {config_name}")
    return results_seed

# ============================================================================
# MAIN ABLATION WITH MULTIPROCESSING
# ============================================================================

def run_ablation_experiment():
    """Run dual-environment ablation with multiprocessing"""
    print("\n" + "="*80)
    print("LIGHTMETA ABLATION STUDY - PARALLEL EXECUTION")
    print("="*80)
    print(f"Configuration: {len(SEEDS)} seeds × {len(TUNING_CONFIGS)} configs × 2 environments")
    print(f"Parallel workers: {NUM_WORKERS}")
    print(f"Estimated time: ~{len(TUNING_CONFIGS) * 4.5:.1f} hours with {NUM_WORKERS} workers\n")
    
    results = {'simple_spread': {}, 'custom_env': {}}
    
    for config_name, config in TUNING_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"Testing: {config_name.upper()}")
        print(f"Description: {config['description']}")
        print(f"{'='*80}")
        
        # Prepare arguments for parallel execution
        args_list = [(config_name, config, seed) for seed in SEEDS]
        
        # Run seeds in parallel
        with mp.Pool(processes=NUM_WORKERS) as pool:
            seed_results = pool.map(run_single_seed, args_list)
        
        # Aggregate results
        results['simple_spread'][config_name] = []
        results['custom_env'][config_name] = []
        
        for seed_result in seed_results:
            if 'simple_spread' in seed_result:
                results['simple_spread'][config_name].append({
                    'seed': seed_result['seed'],
                    **seed_result['simple_spread']
                })
            
            if 'custom_env' in seed_result:
                results['custom_env'][config_name].append({
                    'seed': seed_result['seed'],
                    **seed_result['custom_env']
                })
        
        # Print aggregate for this config
        if results['simple_spread'][config_name]:
            mean_ss = np.mean([r['reward'] for r in results['simple_spread'][config_name]])
            std_ss = np.std([r['reward'] for r in results['simple_spread'][config_name]])
            print(f"\n{config_name} Simple Spread Average: {mean_ss:.2f} ± {std_ss:.2f}")
        
        if results['custom_env'][config_name]:
            mean_custom = np.mean([r['reward'] for r in results['custom_env'][config_name]])
            std_custom = np.std([r['reward'] for r in results['custom_env'][config_name]])
            print(f"{config_name} Custom Env Average: {mean_custom:.2f} ± {std_custom:.2f}")
    
    return results, TUNING_CONFIGS

# ============================================================================
# VISUALIZATION FUNCTIONS (Same as before)
# ============================================================================

def plot_ablation_results(results, configs):
    """Generate dual-environment ablation visualization"""
    
    config_names = list(configs.keys())
    
    means_ss = [np.mean([r['reward'] for r in results['simple_spread'][cfg]]) for cfg in config_names]
    stds_ss = [np.std([r['reward'] for r in results['simple_spread'][cfg]]) for cfg in config_names]
    
    means_custom = [np.mean([r['reward'] for r in results['custom_env'][cfg]]) for cfg in config_names]
    stds_custom = [np.std([r['reward'] for r in results['custom_env'][cfg]]) for cfg in config_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    x_pos = np.arange(len(config_names))
    colors_ss = plt.cm.Blues(np.linspace(0.4, 0.9, len(config_names)))
    colors_custom = plt.cm.Greens(np.linspace(0.4, 0.9, len(config_names)))
    
    bars1 = ax1.barh(x_pos, means_ss, xerr=stds_ss, color=colors_ss, alpha=0.8, 
                     edgecolor='black', linewidth=2, capsize=5)
    
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels([f"{cfg}" for cfg in config_names], fontsize=11)
    ax1.set_xlabel('Average Reward (Mean ± Std)', fontsize=14, fontweight='bold')
    ax1.set_title('Simple Spread Environment', fontsize=16, fontweight='bold')
    ax1.axvline(x=np.mean(means_ss), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(means_ss):.1f}')
    ax1.legend(fontsize=11)
    ax1.grid(True, axis='x', alpha=0.3)
    
    for bar, mean, std in zip(bars1, means_ss, stds_ss):
        ax1.text(mean + std + abs(mean)*0.05, bar.get_y() + bar.get_height()/2,
                f'{mean:.1f}±{std:.1f}',
                va='center', fontsize=9, fontweight='bold')
    
    bars2 = ax2.barh(x_pos, means_custom, xerr=stds_custom, color=colors_custom, alpha=0.8, 
                     edgecolor='black', linewidth=2, capsize=5)
    
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels([f"{cfg}" for cfg in config_names], fontsize=11)
    ax2.set_xlabel('Average Reward (Mean ± Std)', fontsize=14, fontweight='bold')
    ax2.set_title('Custom Environment', fontsize=16, fontweight='bold')
    ax2.axvline(x=np.mean(means_custom), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(means_custom):.1f}')
    ax2.legend(fontsize=11)
    ax2.grid(True, axis='x', alpha=0.3)
    
    for bar, mean, std in zip(bars2, means_custom, stds_custom):
        ax2.text(mean + std + abs(mean)*0.05, bar.get_y() + bar.get_height()/2,
                f'{mean:.1f}±{std:.1f}',
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ablation_study_dual_environment.png', dpi=300, bbox_inches='tight')
    print("\n✓ Ablation plot saved as 'ablation_study_dual_environment.png'")
    plt.show()

def print_ablation_table(results, configs):
    """Print detailed dual-environment results table"""
    print("\n" + "="*140)
    print("ABLATION STUDY RESULTS - DUAL ENVIRONMENT COMPARISON")
    print("="*140)
    print(f"{'Configuration':<22} | {'LayerNorm':<10} | {'Residual':<9} | {'RelBias':<8} | "
          f"{'Simple Spread':<25} | {'Custom Env':<25} | {'Avg Rank':<8}")
    print("-"*140)
    
    ss_means = [(cfg, np.mean([r['reward'] for r in results['simple_spread'][cfg]])) for cfg in configs.keys()]
    custom_means = [(cfg, np.mean([r['reward'] for r in results['custom_env'][cfg]])) for cfg in configs.keys()]
    
    ss_ranks = {cfg: rank for rank, (cfg, _) in enumerate(sorted(ss_means, key=lambda x: x[1], reverse=True), 1)}
    custom_ranks = {cfg: rank for rank, (cfg, _) in enumerate(sorted(custom_means, key=lambda x: x[1], reverse=True), 1)}
    
    avg_ranks = {cfg: (ss_ranks[cfg] + custom_ranks[cfg]) / 2 for cfg in configs.keys()}
    sorted_configs = sorted(avg_ranks.items(), key=lambda x: x[1])
    
    for cfg_name, avg_rank in sorted_configs:
        cfg = configs[cfg_name]
        
        ss_mean = np.mean([r['reward'] for r in results['simple_spread'][cfg_name]])
        ss_std = np.std([r['reward'] for r in results['simple_spread'][cfg_name]])
        
        custom_mean = np.mean([r['reward'] for r in results['custom_env'][cfg_name]])
        custom_std = np.std([r['reward'] for r in results['custom_env'][cfg_name]])
        
        ln_marker = "✓" if cfg['use_layer_norm'] else "✗"
        res_marker = "✓" if cfg['use_residual'] else "✗"
        bias_marker = "✓" if cfg['use_relational_bias'] else "✗"
        
        print(f"{cfg_name:<22} | {ln_marker:^10} | {res_marker:^9} | {bias_marker:^8} | "
              f"{ss_mean:7.2f} ± {ss_std:5.2f} (#{ss_ranks[cfg_name]}) | "
              f"{custom_mean:7.2f} ± {custom_std:5.2f} (#{custom_ranks[cfg_name]}) | "
              f"{avg_rank:6.1f}")
    
    print("="*140)
    
    print("\nCROSS-ENVIRONMENT CONSISTENCY:")
    print("-"*60)
    from scipy.stats import spearmanr
    ss_perf = [np.mean([r['reward'] for r in results['simple_spread'][cfg]]) for cfg in configs.keys()]
    custom_perf = [np.mean([r['reward'] for r in results['custom_env'][cfg]]) for cfg in configs.keys()]
    corr, p_val = spearmanr(ss_perf, custom_perf)
    print(f"  Spearman correlation: ρ={corr:.3f}, p={p_val:.4f}")
    if corr > 0.7:
        print("  ✓ HIGH consistency - architectural choices transfer well")
    elif corr > 0.4:
        print("  ⚠️ MODERATE consistency - some choices are environment-specific")
    else:
        print("  ✗ LOW consistency - highly environment-dependent")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run parallel ablation
    results, configs = run_ablation_experiment()
    
    # Generate visualizations
    plot_ablation_results(results, configs)
    
    # Print detailed table
    print_ablation_table(results, configs)
    
    print("\n" + "="*80)
    print("DUAL-ENVIRONMENT ABLATION STUDY COMPLETE")
    print("="*80)
