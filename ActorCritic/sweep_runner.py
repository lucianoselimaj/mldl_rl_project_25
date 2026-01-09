import wandb
import yaml
from train_actor_critic import train_actor_critic

SWEEP_CONFIG_PATH = "sweep_config_ac.yaml"


def sweep_train():
    # 1. Load the sweep configuration
    with open(SWEEP_CONFIG_PATH) as f:
        sweep_config = yaml.safe_load(f)

    # 2. Create the sweep on WandB servers
    sweep_id = wandb.sweep(sweep=sweep_config, project="actor-critic-hopper")

    # 3. Define the function that the agent will execute
    def run_sweep():
        # Initialize the run
        run = wandb.init(project="actor-critic-hopper")
        cfg = wandb.config

        # Generate a unique name for this run based on params
        ac_type = "AC" if cfg.actor_critic else "REINFORCE"
        run.name = f"{ac_type}_Base{cfg.baseline}_Seed{cfg.seed}"

        # Call your training function
        train_actor_critic(config=cfg, run_name=run.name)

    # 4. Start the agent
    wandb.agent(sweep_id, function=run_sweep)


if __name__ == "__main__":
    sweep_train()