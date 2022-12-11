import wandb

class StatLogger():
    def __init__(self, type):
        self.type = type
        self.step = 0
    
    def log(self, log):
        wandb.log({self.type + "/loss" : log["loss"]}, step=self.step)

        accuracy_log = {}
        for key in log["accuracy"]:
            accuracy_log[self.type + "/accuracy/" + key] = log["accuracy"][key]
        wandb.log(accuracy_log, step=self.step)
        
        f1_log = {}
        for key in log["f1_score"]:
            f1_log[self.type + "/f1_score/" + key] = log["f1_score"][key]
        wandb.log(f1_log, step=self.step)

        if type == "train":
            for exposure_bias_type in log["exposure_bias"]:
                accuracy_log = {}
                for key in log[exposure_bias_type]["accuracy"]:
                    accuracy_log[
                        self.type + f"/{exposure_bias_type}/" + "/accuracy/" + key
                    ] = log["accuracy"][exposure_bias_type][key]
                wandb.log(accuracy_log, step=self.step)

        self.step += 1



