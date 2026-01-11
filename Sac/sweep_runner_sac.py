import wandb, yaml
from train_sac import train_sac

SWEEP_CONFIG_PATH = "sweep_config_sac.yaml"

def sweep_train():
    with open(SWEEP_CONFIG_PATH) as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project="sac-hopper")

    def run_sweep():
        run = wandb.init(project="sac-hopper", sync_tensorboard=True)
        cfg = wandb.config
        run.name = f"SAC_seed{cfg.seed}_basic"
        train_sac(config=dict(cfg), run_name=run.name)
        wandb.finish()

    wandb.agent(sweep_id, function=run_sweep)

if __name__ == "__main__":
    sweep_train()
