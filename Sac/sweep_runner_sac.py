import os
import wandb, yaml
from train_sac import train_sac

SWEEP_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "sweep_config_sac.yaml")
BETA_CONFIG_PATH  = os.path.join(os.path.dirname(__file__), "adv_beta_config.yaml")

def sweep_train():
    with open(SWEEP_CONFIG_PATH) as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project="sac-hopper")

    def run_sweep():
        run = wandb.init(project="sac-hopper", sync_tensorboard=True)
        cfg = wandb.config

        run.name = f"SAC_seed{cfg.seed}_{cfg.dr_method}"

        beta_config = {}
        if cfg.dr_method == "adv_beta":
            with open(BETA_CONFIG_PATH) as f:
                beta_config = yaml.safe_load(f)
            run.config.update(beta_config, allow_val_change=True)

        train_sac(config=dict(cfg), run_name=run.name, beta_config=beta_config)
        wandb.finish()

    wandb.agent(sweep_id, function=run_sweep)

if __name__ == "__main__":
    sweep_train()
