import torch


class EarlyStopper:
    def __init__(self, patience, model, file_name):
        self.best_eval_loss = float("inf")
        self.early_stop_counter = 0
        self.end_flag = False
        self.PATIENCE = patience
        self.file_name = file_name
        self.model = model
        pass

    def check(self, eval_loss):
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.early_stop_counter = 0
            print(f" └ Updated best eval loss : {self.best_eval_loss:.4f}")
            torch.save(self.model.state_dict(), self.file_name + ".pth")
            return False
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.PATIENCE:
                print(f"Early stop!! best_eval_loss = {self.best_eval_loss:.4f}")
                self.end_flag = True
                return True
            else:
                # print("Not early stop")
                return False

    def state_dict(self):
        return {
            "best_eval_loss": self.best_eval_loss,
            "early_stop_counter": self.early_stop_counter,
            "end_flag": self.end_flag,
        }

    def load_state_dict(self, state_dict):
        self.best_eval_loss = state_dict["best_eval_loss"]
        self.early_stop_counter = state_dict["early_stop_counter"]
        self.end_flag = state_dict["end_flag"]

        return
