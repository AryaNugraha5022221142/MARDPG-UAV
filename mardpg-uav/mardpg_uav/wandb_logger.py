import wandb

class WandbLogger:
    def __init__(self, use_wandb, project, config, name):
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project=project, config=config, name=name)

    def log(self, data, step=None):
        if self.use_wandb:
            wandb.log(data, step=step)

    def finish(self):
        if self.use_wandb:
            wandb.finish()
