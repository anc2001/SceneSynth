import wandb
import numpy as np

class StatLogger():
    def __init__(self, type):
        self.type = type
        self.step = 0
        self.stats = {
            "loss": [],
            "accuracy": {
                "structure" : [],
                "type" : [],
                "object" : [],
                "direction" : [],
                "total" : []                
            },
            "f1_score": {
                "structure" : [],
                "type" : [],
                "object" : [],
                "direction" : []
            }
        }
    
    def log(self, log, accum=False):
        if accum:
            self.stats["loss"].append(log["loss"])
        wandb.log({self.type + "/loss" : log["loss"]}, step=self.step)

        accuracy_log = {}
        for key in log["accuracy"]:
            if accum:
                self.stats["accuracy"][key].append(log["accuracy"][key])
            accuracy_log[self.type + "/accuracy/" + key] = log["accuracy"][key]
        wandb.log(accuracy_log, step=self.step)
        
        f1_log = {}
        for key in log["f1_score"]:
            if accum:
                self.stats["f1_score"][key].append(log["f1_score"][key])
            f1_log[self.type + "/f1_score/" + key] = log["f1_score"][key]
        wandb.log(f1_log, step=self.step)

        if self.type == "train":
            for exposure_bias_type in log["exposure_bias"]:
                accuracy_log = {}
                exposure_bias_log = log["exposure_bias"][exposure_bias_type]
                for key in exposure_bias_log:
                    accuracy_log[
                        self.type + f"/{exposure_bias_type}/accuracy/" + key
                    ] = exposure_bias_log[key]
                wandb.log(accuracy_log, step=self.step)

        self.step += 1
    
    def get_summary(self):
        average_loss = np.mean(self.stats["loss"])
        accuracy_metric = {}
        for key in self.stats["accuracy"]:
            accuracy_metric[key] = np.mean(self.stats["accuracy"][key])
        f1_metric = {}
        for key in self.stats["f1_score"]:
            f1_metric[key] = np.mean(self.stats["f1_score"][key])

        summary = {
            "loss" : average_loss,
            "accuracy" : accuracy_metric,
            "f1_score" : f1_metric
        }
        return summary



