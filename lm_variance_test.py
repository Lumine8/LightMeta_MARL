import copy
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.mpe import simple_spread_v3

# ============================================================================
# CONFIGURATION - Now includes EXACT ORIGINAL config
# ============================================================================

# TEST MODES
TEST_BASELINE = True       # Original settings (rank=8)
TEST_EXACT_ORIGINAL = True # ← NEW: Exact match to paper code (rank=4)
TEST_REDUCED_ADAPT = True  # Fewer adaptation steps

SEEDS = [42, 123, 888, 2024, 7]
MAX_AGENTS = 5
META_ITERATIONS = 500

# Baseline settings (rank=8, might not match paper)
BASELINE_CONFIG = {
    'meta_lr': 0.0003,
    'adapt_lr': 0.001,
    'adapt_steps': 40,
    'lorasa_rank': 8,
    'grad_clip': 1.0,
    'entropy_coef': 0.01,
    'use_experience_buffer': False  # Old online adaptation
}

# EXACT ORIGINAL paper settings
EXACT_ORIGINAL_CONFIG = {
    'meta_lr': 0.0003,
    'adapt_lr': 0.001,      # Matches original
    'adapt_steps': 40,      # Matches original
    'lorasa_rank': 4,       # ← KEY: rank=4 as in original
    'grad_clip': 1.0,
    'entropy_coef': 0.01,
    'use_experience_buffer': True  # ← NEW: Use experience buffer method
}

# Reduced adaptation (for comparison)
REDUCED_ADAPT_CONFIG = {
    'meta_lr': 0.0003,
    'adapt_lr': 0.0003,
    'adapt_steps': 15,
    'lorasa_rank': 4,
    'grad_clip': 0.5,
    'entropy_coef': 0.01,
    'use_experience_buffer': False
}

# ============================================================================
# ENVIRONMENT
# ============================================================================

class SimpleSpreadWrapper(gym.Env):
    @staticmethod
    def get_max_obs_dim(max_agents):
        temp_env = simple_spread_v3.parallel_env(N=max_agents)
        max_dim = temp_env.observation_space("agent_0").shape[0]
        temp_env.close()
        return max_dim
    
    def __init__(self, num_agents=5, seed=None):
        if num_agents is None:
            num_agents = np.random.choice([2, 3, 4, 5])
        
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
# LoRASA Layer
# ============================================================================

class LoRASALayer(nn.Module):
    def __init__(self, linear_layer, rank=4):
        super().__init__()
        self.linear = linear_layer
        in_features, out_features = linear_layer.in_features, linear_layer.out_features
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

def inject_lorasa(model, rank=4):
    """Recursively inject LoRASA into all Linear layers"""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRASALayer(module, rank=rank))
        elif len(list(module.children())) > 0:
            inject_lorasa(module, rank=rank)
    return model

# ============================================================================
# LightMeta Policy
# ============================================================================

class LightMetaPolicy(nn.Module):
    def __init__(self, agent_obs_dim, num_actions):
        super().__init__()
        self.agent_dim = agent_obs_dim
        self.num_actions = num_actions
        self.d_model = 64
        
        self.input_proj = nn.Linear(self.agent_dim, self.d_model)
        self.key_transform = nn.Linear(self.d_model, self.d_model)
        self.query_transform = nn.Linear(self.d_model, self.d_model)
        self.value_transform = nn.Linear(self.d_model, self.d_model)
        self.fc_out = nn.Linear(self.d_model, self.d_model)
        
        self.action_head = nn.Linear(self.d_model, MAX_AGENTS * self.num_actions)
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
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
        attention = torch.softmax(attention, dim=-1)
        context = torch.matmul(attention, values)
        context = self.fc_out(context)
        
        agent_mask = (agents.abs().sum(dim=-1, keepdim=True) > 0.01).float()
        pooled = (context * agent_mask).sum(dim=1) / (agent_mask.sum(dim=1) + 1e-8)
        
        action_logits = self.action_head(pooled).view(batch_size, MAX_AGENTS, self.num_actions)
        value = self.value_head(pooled)
        return action_logits, value

# ============================================================================
# Training Function
# ============================================================================

def train_light_meta(model, env_fn, config, meta_iterations=500, verbose=False):
    """Meta-train LightMeta"""
    optimizer = optim.Adam(model.parameters(), lr=config['meta_lr'])
    
    for iteration in range(meta_iterations):
        num_agents = np.random.choice([2, 3, 4, 5])
        env = env_fn(num_agents=num_agents)
        obs, _ = env.reset()
        
        log_probs, rewards, values = [], [], []
        for _ in range(128):  # Inner rollouts
            with torch.no_grad():
                action_logits, value = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                dist = torch.distributions.Categorical(logits=action_logits[:, :num_agents, :])
                actions = dist.sample()
            
            next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
            
            # Re-compute for gradient
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
        
        # Compute returns
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + 0.99 * discounted_reward
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.cat(values).squeeze()
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        policy_loss = -(torch.stack(log_probs) * advantages).mean()
        value_loss = nn.MSELoss()(values, returns)
        
        loss = policy_loss + 0.5 * value_loss - config['entropy_coef'] * dist.entropy().mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip'])
        optimizer.step()
        
        if verbose and iteration % 100 == 0:
            print(f"  Iteration {iteration}/{meta_iterations}, Loss: {loss.item():.4f}")
    
    return model

# ============================================================================
# Adaptation Functions - ORIGINAL vs NEW
# ============================================================================

def adapt_with_lorasa_original(model, env_fn, config):
    """
    EXACT MATCH to original paper code.
    Collects 5 episodes of experiences FIRST using original model,
    then adapts LoRASA on batches from this replay buffer.
    """
    # Deep copy and inject LoRASA
    tuned_model = copy.deepcopy(model)
    inject_lorasa(tuned_model, rank=config['lorasa_rank'])
    
    # Freeze base parameters
    for name, param in tuned_model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    
    # Optimizer for LoRASA parameters only
    lorasa_params = [p for p in tuned_model.parameters() if p.requires_grad]
    optimizer = optim.Adam(lorasa_params, lr=config['adapt_lr'])
    
    env = env_fn(num_agents=5)
    
    # STEP 1: Collect experiences using ORIGINAL model (5 episodes)
    experiences = []
    for episode in range(5):
        obs, _ = env.reset()
        done = False
        
        while not done:
            # Use ORIGINAL model (not tuned_model!)
            with torch.no_grad():
                action_logits, _ = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                dist = torch.distributions.Categorical(logits=action_logits[:, :5, :])
                actions = dist.sample()
            
            next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
            experiences.append((obs, actions.numpy().flatten(), reward, next_obs))
            obs = next_obs
            done = terminated
    
    # STEP 2: Adapt LoRASA on batches from experience buffer
    for step in range(config['adapt_steps']):
        if len(experiences) > 64:
            batch = random.sample(experiences, 64)
            batch_obs = torch.tensor(np.array([s[0] for s in batch]), dtype=torch.float32)
            batch_actions = torch.tensor(np.array([s[1] for s in batch]), dtype=torch.long)
            batch_rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32)
            
            # Forward pass with tuned model
            action_logits, _ = tuned_model(batch_obs)
            dist = torch.distributions.Categorical(logits=action_logits[:, :5, :])
            log_probs = dist.log_prob(batch_actions)
            
            # Simple policy gradient loss (matches original exactly)
            loss = -(log_probs * batch_rewards.unsqueeze(1)).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lorasa_params, max_norm=config['grad_clip'])
            optimizer.step()
    
    return tuned_model

def adapt_with_lorasa_online(model, env_fn, config):
    """
    Online adaptation (old method).
    For comparison with experience buffer method.
    """
    model_copy = copy.deepcopy(model)
    inject_lorasa(model_copy, rank=config['lorasa_rank'])
    
    for name, param in model_copy.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    
    lorasa_params = [p for p in model_copy.parameters() if p.requires_grad]
    optimizer = optim.Adam(lorasa_params, lr=config['adapt_lr'])
    env = env_fn(num_agents=5)
    
    for step in range(config['adapt_steps']):
        obs, _ = env.reset()
        log_probs, rewards, values = [], [], []
        
        for _ in range(20):
            action_logits, value = model_copy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            dist = torch.distributions.Categorical(logits=action_logits[:, :5, :])
            actions = dist.sample()
            
            next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
            
            log_probs.append(dist.log_prob(actions).mean())
            rewards.append(reward)
            values.append(value)
            
            obs = next_obs
            if terminated:
                break
        
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
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        policy_loss = -(torch.stack(log_probs) * advantages).mean()
        value_loss = nn.MSELoss()(values, returns)
        loss = policy_loss + 0.5 * value_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lorasa_params, max_norm=config['grad_clip'])
        optimizer.step()
    
    return model_copy

# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model, env_fn, episodes=20):
    """Evaluate model"""
    total_rewards = []
    
    for ep in range(episodes):
        env = env_fn(num_agents=5, seed=1000 + ep)
        obs, _ = env.reset()
        episode_reward = 0
        
        for _ in range(100):
            with torch.no_grad():
                action_logits, _ = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                actions = torch.argmax(action_logits[:, :5, :], dim=-1).numpy().flatten()
            
            obs, reward, terminated, _, _ = env.step(actions)
            episode_reward += reward
            if terminated:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards), np.std(total_rewards), total_rewards

# ============================================================================
# Main Test
# ============================================================================

def run_test_config(config_name, config, seeds):
    """Run test for a specific configuration"""
    print(f"\n{'='*70}")
    print(f"Testing Configuration: {config_name}")
    print(f"{'='*70}")
    print(f"Settings: meta_lr={config['meta_lr']}, adapt_lr={config['adapt_lr']}, "
          f"adapt_steps={config['adapt_steps']}, lorasa_rank={config['lorasa_rank']}, "
          f"experience_buffer={config['use_experience_buffer']}")
    
    max_obs_dim = SimpleSpreadWrapper.get_max_obs_dim(MAX_AGENTS)
    action_dim = SimpleSpreadWrapper(num_agents=2).action_dim
    env_fn = lambda num_agents=None, seed=None: SimpleSpreadWrapper(num_agents=num_agents, seed=seed)
    
    all_zero_shot = []
    all_adapted = []
    all_improvements = []
    
    for seed in seeds:
        print(f"\n--- Seed: {seed} ---")
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        model = LightMetaPolicy(max_obs_dim, action_dim)
        
        # Meta-train
        print("  Meta-training...")
        model = train_light_meta(model, env_fn, config, meta_iterations=META_ITERATIONS, verbose=False)
        
        # Evaluate zero-shot
        reward_zs, std_zs, _ = evaluate(model, env_fn, episodes=20)
        all_zero_shot.append(reward_zs)
        print(f"  Zero-shot: {reward_zs:.2f} ± {std_zs:.2f}")
        
        # Adapt (choose method based on config)
        print(f"  Adapting with LoRASA ({'experience buffer' if config['use_experience_buffer'] else 'online'})...")
        if config['use_experience_buffer']:
            model_adapted = adapt_with_lorasa_original(model, env_fn, config)
        else:
            model_adapted = adapt_with_lorasa_online(model, env_fn, config)
        
        # Evaluate adapted
        reward_adapted, std_adapted, _ = evaluate(model_adapted, env_fn, episodes=20)
        all_adapted.append(reward_adapted)
        print(f"  Adapted:   {reward_adapted:.2f} ± {std_adapted:.2f}")
        
        # Check improvement
        improvement = reward_adapted - reward_zs
        all_improvements.append(improvement)
        status = '✓ BETTER' if improvement > 0 else '✗ WORSE'
        print(f"  Improvement: {improvement:.2f} ({status})")
    
    # Aggregate statistics
    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS: {config_name}")
    print(f"{'='*70}")
    print(f"Zero-shot:  {np.mean(all_zero_shot):7.2f} ± {np.std(all_zero_shot):6.2f} "
          f"(CV: {(np.std(all_zero_shot)/abs(np.mean(all_zero_shot))*100):5.1f}%)")
    print(f"Adapted:    {np.mean(all_adapted):7.2f} ± {np.std(all_adapted):6.2f} "
          f"(CV: {(np.std(all_adapted)/abs(np.mean(all_adapted))*100):5.1f}%)")
    print(f"Improvement:{np.mean(all_improvements):7.2f} ± {np.std(all_improvements):6.2f}")
    
    positive_improvements = sum(1 for imp in all_improvements if imp > 0)
    print(f"\nAdaptation Success Rate: {positive_improvements}/{len(seeds)} seeds ({positive_improvements/len(seeds)*100:.0f}%)")
    
    cv_zs = np.std(all_zero_shot) / abs(np.mean(all_zero_shot))
    cv_adapted = np.std(all_adapted) / abs(np.mean(all_adapted))
    
    print(f"\nVariance Analysis:")
    print(f"  Zero-shot CV:  {cv_zs*100:5.1f}% {'✓ Good' if cv_zs < 0.5 else '⚠️ High' if cv_zs < 1.0 else '❌ Critical'}")
    print(f"  Adapted CV:    {cv_adapted*100:5.1f}% {'✓ Good' if cv_adapted < 0.5 else '⚠️ High' if cv_adapted < 1.0 else '❌ Critical'}")
    
    return {
        'config_name': config_name,
        'zero_shot': all_zero_shot,
        'adapted': all_adapted,
        'improvements': all_improvements,
        'cv_zs': cv_zs,
        'cv_adapted': cv_adapted
    }

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LIGHTMETA ADAPTATION TEST - EXACT ORIGINAL vs VARIANTS")
    print("="*70)
    print(f"Testing with {len(SEEDS)} seeds: {SEEDS}")
    print(f"Meta-iterations: {META_ITERATIONS}")
    
    results = []
    
    if TEST_EXACT_ORIGINAL:
        results.append(run_test_config("EXACT_ORIGINAL", EXACT_ORIGINAL_CONFIG, SEEDS))
    
    if TEST_BASELINE:
        results.append(run_test_config("BASELINE", BASELINE_CONFIG, SEEDS))
    
    if TEST_REDUCED_ADAPT:
        results.append(run_test_config("REDUCED_ADAPT", REDUCED_ADAPT_CONFIG, SEEDS))
    
    # Compare all configurations
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("CONFIGURATION COMPARISON")
        print(f"{'='*70}")
        print(f"{'Config':<20} | {'Adapted Mean':<12} | {'Improvement':<12} | {'Success Rate'}")
        print(f"{'-'*70}")
        
        for r in results:
            adapted_mean = np.mean(r['adapted'])
            improvement = np.mean(r['improvements'])
            success_rate = sum(1 for imp in r['improvements'] if imp > 0) / len(r['improvements']) * 100
            
            print(f"{r['config_name']:<20} | {adapted_mean:>11.2f} | {improvement:>11.2f} | {success_rate:>5.0f}%")
        
        best_improvement = max(results, key=lambda x: np.mean(x['improvements']))
        print(f"\n✓ BEST CONFIGURATION (highest improvement): {best_improvement['config_name']}")
        print(f"  Mean Improvement: +{np.mean(best_improvement['improvements']):.2f}")
        print(f"  Adapted Performance: {np.mean(best_improvement['adapted']):.2f}")
    
    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")
