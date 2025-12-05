import wandb
from train_sb3 import train_sac
import yaml

SWEEP_CONFIG_PATH = "sweep_config.yaml"

def sweep_train():
    # Create sweep on WandB
    with open(SWEEP_CONFIG_PATH) as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project="sac")

    # Initialize WandB
    def run_sweep():
        run = wandb.init(
            project="sac",
            sync_tensorboard=True,
            monitor_gym=False,  # Enable/disable video recording
            save_code=False,
        )

        config = wandb.config  
        run.name=f"sac_seed_{wandb.config.seed}_id_{wandb.util.generate_id()}"  # Run's name on WandB

        train_sac(config, run.name)

        run.finish()

    # Reproduce training metrics of sweep configurations on WandB
    wandb.agent(sweep_id, function=run_sweep, count=None) 

if __name__ == "__main__":
    # Train sweep configurations
    sweep_train()