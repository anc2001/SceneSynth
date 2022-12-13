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

        self.step += 1
    
    def accumulate(self, log):
        self.stats["loss"].append(log["loss"])

        for key in log["accuracy"]:
            self.stats["accuracy"][key].append(log["accuracy"][key])
        
        for key in log["f1_score"]:
            self.stats["f1_score"][key].append(log["f1_score"][key])

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



