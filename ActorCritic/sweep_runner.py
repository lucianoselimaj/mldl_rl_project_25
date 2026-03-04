import os
import wandb
import yaml
from train_actor_critic import train_actor_critic

SWEEP_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "sweep_config_ac.yaml")


def sweep_train():
    # 1. Load the sweep configuration
    with open(SWEEP_CONFIG_PATH) as f:
        sweep_config = yaml.safe_load(f)

    # 2. Create the sweep on WandB servers
    sweep_id = wandb.sweep(sweep=sweep_config, project="ac-reinforce-hopper-seed")

    # 3. Define the function that the agent will execute
    def run_sweep():
        # Initialize the run
        run = wandb.init(project="ac-reinforce-hopper-seed")
        cfg = wandb.config

        # Generate a unique name for this run based on params
        algo = "AC" if cfg.actor_critic else f"REINFORCE_Base{cfg.baseline}"
        run.name = f"{algo}_seed{cfg.seed}_{cfg.dr_method}"

        # Call your training function
        train_actor_critic(config=cfg, run_name=run.name)

    # 4. Start the agent
    wandb.agent(sweep_id, function=run_sweep)


if __name__ == "__main__":
    sweep_train()