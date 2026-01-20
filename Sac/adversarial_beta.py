import numpy as np

class AdversarialBeta:
    def __init__(self, nominal_masses, 
                 limit_percentage=0.3,   # +/- 30%
                 buffer_size=300,        # Optimized for approximately 3000 ep in total
                 warmup_episodes=750,
                 mix_ratio=0.5,          # 50% Uniform, 50% Adv beta 
                 tau=0.1,                # Soft update rate
                 max_alpha_beta=80.0,    # Clipping to avoid overfitting 
                 seed=42):
        
        self.rng = np.random.default_rng(seed)
        
        # Configuration of parameters
        self.nominal = np.array(nominal_masses)
        self.n_dims = len(self.nominal)
        
        # Bounds
        self.lower = self.nominal * (1.0 - limit_percentage)
        self.upper = self.nominal * (1.0 + limit_percentage)
        self.range = self.upper - self.lower
        
        # Buffer (FIFO)
        self.buffer_size = buffer_size
        self.buffer_params = []
        self.buffer_rewards = []
        
        # Beta parameters initialized to 1.0
        self.alphas = np.ones(self.n_dims)
        self.betas = np.ones(self.n_dims)
        
        # Hyperparameters
        self.mix_ratio = mix_ratio
        self.tau = tau
        self.max_param_val = max_alpha_beta
        self.failure_percentile = 20 # Percentile of failures
        self.warmup_episodes = warmup_episodes

        self.episode_count = 0

    def normalize(self, params):
        """Map from [Lower, Upper] to [0, 1]"""
        return (params - self.lower) / (self.range + 1e-8)

    def denormalize(self, normalized_params):
        """Map from [0, 1] to [Lower, Upper]"""
        return normalized_params * self.range + self.lower

    def add_experience(self, params, reward):
        """Add episode to the buffer and drop the ones over the fixed length"""
        self.buffer_params.append(params)
        self.buffer_rewards.append(reward)
        self.episode_count += 1
        
        if len(self.buffer_params) > self.buffer_size:
            self.buffer_params.pop(0)
            self.buffer_rewards.pop(0)

    def fit_model(self):
        """
        Update Beta with method of moments fitted on failures
        """

        if self.episode_count < self.warmup_episodes:
            return

        # Bottom self.failure_percentile% rewards are failures
        threshold = np.percentile(self.buffer_rewards, self.failure_percentile)
        fail_indices = [i for i, r in enumerate(self.buffer_rewards) if r <= threshold]
        
        if len(fail_indices) < 10: 
            return # Troppo pochi dati per fittare una distribuzione
            
        fail_data = np.array(self.buffer_params)[fail_indices]

        # Update for each mass
        for dim in range(self.n_dims):
            # Normalize 
            data_dim = fail_data[:, dim]
            range_val = self.range[dim] + 1e-8
            norm_data = (data_dim - self.lower[dim]) / range_val
            
            # Avoid log(0)
            norm_data = np.clip(norm_data, 0.01, 0.99)
            
            # Metod of moments
            mu = np.mean(norm_data)
            var = np.var(norm_data)
            
            if var < mu * (1 - mu):
                common_factor = ((mu * (1 - mu)) / (var + 1e-6)) - 1
                new_alpha = mu * common_factor
                new_beta = (1 - mu) * common_factor
            else:
                # If the variance is too large, return to uniform
                new_alpha, new_beta = 1.0, 1.0
            
            # Clip to ensure reasonable shapes
            # Min 1.0 to avoid U shape
            # Max e.g. 80.0 to avoid overfitting of beta
            new_alpha = np.clip(new_alpha, 1.0, self.max_param_val)
            new_beta  = np.clip(new_beta,  1.0, self.max_param_val)
            
            # Soft update of betas
            self.alphas[dim] = (1 - self.tau) * self.alphas[dim] + self.tau * new_alpha
            self.betas[dim]  = (1 - self.tau) * self.betas[dim]  + self.tau * new_beta

    def sample_task(self):
        """Sample parameters"""

        # Uniform
        # Mix ratio and warmup phase
        if self.episode_count < self.warmup_episodes:
            return self.rng.uniform(self.lower, self.upper)
        
        if self.rng.random() > self.mix_ratio:
            return self.rng.uniform(self.lower, self.upper)
        
        # Beta
        try:
            samples = self.rng.beta(self.alphas, self.betas)
            return self.denormalize(samples)
        except Exception:
            # Uniform in case of exceptions
            return self.rng.uniform(self.lower, self.upper)

    def get_diagnostics(self):
        """
        Store values of alfas to understand the behavior of betas during training 
        These values will be printed
        """
        return {
            "ep": self.episode_count,
            "warmup": self.episode_count < self.warmup_episodes,
            "alphas": np.round(self.alphas, 2),
            "betas": np.round(self.betas, 2)
        }