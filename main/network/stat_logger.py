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
        }
    
    def log(self, log):
        wandb.log({self.type + "/loss" : log["loss"]}, step=self.step)

        accuracy_log = {}
        for key in log["accuracy"]:
            accuracy_log[self.type + "/accuracy/" + key] = log["accuracy"][key]
        wandb.log(accuracy_log, step=self.step)

        self.step += 1
    
    def accumulate(self, log):
        self.stats["loss"].append(log["loss"])

        for key in log["accuracy"]:
            self.stats["accuracy"][key].append(log["accuracy"][key])

        self.step += 1

    def get_summary(self):
        summary = dict()
        summary["loss"] = np.mean(self.stats["loss"])
        
        for key in self.stats["accuracy"]:
            summary[f"accuracy/{key}"] = np.mean(self.stats["accuracy"][key])

        return summary



