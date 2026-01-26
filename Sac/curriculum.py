import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class FailureGMMCurriculum:
    def __init__(self, nominal_masses, buffer_size=1000, warmup_episodes=50, n_components=2, limit_percentage=0.3, seed=42):
        """
        nominal_masses: masses array 
        limit_percentage:  +/- 30%
        """
        self.seed = seed
        
        self.nominal_masses = np.array(nominal_masses)
        self.n_params = len(nominal_masses)
        
        # Bounds in domain parameters
        self.limit_percentage = limit_percentage
        self.lower_bound = self.nominal_masses * (1.0 - self.limit_percentage)
        self.upper_bound = self.nominal_masses * (1.0 + self.limit_percentage)
        
        # Buffer
        self.buffer_size = buffer_size
        self.history_params = []
        self.history_rewards = []
        
        # GMM model
        self.n_components = n_components
        self.gmm = None
        self.warmup_episodes = warmup_episodes
        self.episode_count = 0
        
        # Hyperparameters
        self.alpha_scaling = 50.0  
        self.mix_ratio = 0.8       
        self.failure_percentile = 30 

    def add_experience(self, params, reward):
        self.history_params.append(params)
        self.history_rewards.append(reward)
        self.episode_count += 1
        
        if len(self.history_params) > self.buffer_size:
            self.history_params.pop(0)
            self.history_rewards.pop(0)

    def fit_model(self):
        if len(self.history_rewards) < self.warmup_episodes:
            return 

        threshold = np.percentile(self.history_rewards, self.failure_percentile)
        failure_indices = [i for i, r in enumerate(self.history_rewards) if r <= threshold]
        
        if len(failure_indices) < 10: return 
        
        failure_data = np.array(self.history_params)[failure_indices]
        failure_rewards = np.array(self.history_rewards)[failure_indices]

        try:
            self.gmm = GaussianMixture(n_components=self.n_components, 
                                       covariance_type='full', 
                                       reg_covar=1e-3,
                                       random_state=self.seed)
            self.gmm.fit(failure_data)

            labels = self.gmm.predict(failure_data)
            new_covariances = []
            
            for i in range(self.n_components):
                sigma = self.gmm.covariances_[i]
                cluster_rews = failure_rewards[labels == i]
                avg_rew = np.mean(cluster_rews) if len(cluster_rews) > 0 else 0.0
                
                # Inverse Reward Scaling
                scaling_factor = 1.0 + (self.alpha_scaling / (max(avg_rew, 0.0) + 1.0))
                scaling_factor = np.clip(scaling_factor, 1.0, 5.0) 
                
                new_covariances.append(sigma * scaling_factor)
                
            self.gmm.covariances_ = np.array(new_covariances)
            self.gmm.precisions_cholesky_ = np.linalg.cholesky(
                np.linalg.inv(self.gmm.covariances_)
            )
        except Exception as e:
            print(f"Curriculum  Fitting failed: {e}")
            self.gmm = None

    def sample_task(self):
        """
        Rejection sampling + Safety fallback.

        """
        is_warmup = self.episode_count < self.warmup_episodes
        do_uniform = np.random.random() > self.mix_ratio
        
        # First case: uniform or warmup 
        if is_warmup or do_uniform or self.gmm is None:
            return np.random.uniform(self.lower_bound, self.upper_bound)

        # Second case: GMM with rejection sampling
        try:
            candidates, _ = self.gmm.sample(n_samples=500)
            
            valid_mask = np.all((candidates >= self.lower_bound) & 
                                (candidates <= self.upper_bound), axis=1)
            
            if np.any(valid_mask):
                return candidates[valid_mask][0]
            
            # Third case: fallback
            return np.clip(candidates[0], self.lower_bound, self.upper_bound)
            
        except Exception:
            return np.random.uniform(self.lower_bound, self.upper_bound)

    def debug_plot(self):
        if self.gmm is None:
            print("Model is not fitted yet.")
            return

        data = np.array(self.history_params)
        rewards = np.array(self.history_rewards)
        
        plt.figure(figsize=(8, 6))
        
        plt.hlines([self.lower_bound[1], self.upper_bound[1]], 
                   self.lower_bound[0], self.upper_bound[0], colors='k', linestyles='--')
        plt.vlines([self.lower_bound[0], self.upper_bound[0]], 
                   self.lower_bound[1], self.upper_bound[1], colors='k', linestyles='--', label='Constraint +/- 30%')

        sc = plt.scatter(data[:, 0], data[:, 1], c=rewards, cmap='RdYlGn', s=10, alpha=0.6)
        plt.colorbar(sc, label='Reward')
        
        ax = plt.gca()
        if self.gmm is not None:
             for i in range(self.n_components):
                mean = self.gmm.means_[i]
                cov = self.gmm.covariances_[i]
                vals, vecs = np.linalg.eigh(cov[:2, :2])
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = 2 * np.sqrt(vals)
                ell = Ellipse(xy=mean[:2], width=width*2, height=height*2, 
                              edgecolor='blue', facecolor='none', lw=2, linestyle='-')
                ax.add_patch(ell)

        plt.title(f"GMM Adaptation (Ep {self.episode_count})")
        plt.xlabel("Mass Param 1")
        plt.ylabel("Mass Param 2")
        plt.legend()
        plt.show()