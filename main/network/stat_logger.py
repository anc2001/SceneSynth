import wandb
import numpy as np

class StatLogger():
    def __init__(self, type):
        self.type = type
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
        self.stats["loss"].append(log["loss"])
        for key in log["accuracy"]:
            self.stats["accuracy"][key].append(log["accuracy"][key])
        
        for key in log["f1_score"]:
            self.stats["f1_score"][key].append(log["f1_score"][key])
    
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

    def log_graphs(self):
        xs = list(range(len(self.stats["loss"])))

        accuracy_stats = self.stats["accuracy"]
        accuracy_keys = list(accuracy_stats.keys())
        accuracy_stats_y_vals = [accuracy_stats[key] for key in accuracy_keys]
        accuracy_graph = wandb.plot.line_series(
            xs=xs, 
            ys=accuracy_stats_y_vals,
            keys=accuracy_keys,
            title= f"{self.type} accuracies",
            xname="Batch Step"
        )

        f1_stats = self.stats["f1_score"]
        f1_keys = list(f1_stats.keys())
        f1_stats_y_vals = [f1_stats[key] for key in f1_keys]
        f1_graph = wandb.plot.line_series(
            xs=xs, 
            ys=f1_stats_y_vals,
            keys=f1_keys,
            title= f"{self.type} F1 Scores",
            xname="Batch Step"
        )

        data = [[x, y] for (x, y) in zip(xs, self.stats["loss"])]
        table = wandb.Table(data=data, columns = ["Batch Step", "Loss"])
        loss_graph = wandb.plot.line(table, "Batch Step", "Loss", title="Loss Graph")
        
        wandb.log({f"{self.type}/loss" : loss_graph})
        wandb.log({f"{self.type}/accuracy" : accuracy_graph})
        wandb.log({f"{self.type}/f1_graph" : f1_graph})



