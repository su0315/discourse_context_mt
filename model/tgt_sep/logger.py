from transformers import TrainerCallback, IntervalStrategy
import collections

class CustomLoggerCallback(TrainerCallback):
    def __init__(self):
        self.training_tracker = None
        self.prediction_bar = None
        self._force_next_update = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.first_column = "Epoch" if args.evaluation_strategy == IntervalStrategy.EPOCH else "Step"
        self.training_loss = 0
        self.last_log = 0
        column_names = [self.first_column] + ["Training Loss"]
        if args.evaluation_strategy != IntervalStrategy.NO:
            column_names.append("Validation Loss")
        

    def on_step_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if int(state.epoch) == state.epoch else f"{state.epoch:.2f}"
        self.training_tracker.update(
            state.global_step + 1,
            comment=f"Epoch {epoch}/{state.num_train_epochs}",
            force_update=self._force_next_update,
        )
        self._force_next_update = False

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if not isinstance(eval_dataloader.dataset, collections.abc.Sized):
            return
        if self.prediction_bar is None:
            if self.training_tracker is not None:
                self.prediction_bar = self.training_tracker.add_child(len(eval_dataloader))
            self.prediction_bar.update(1)
        else:
            self.prediction_bar.update(self.prediction_bar.value + 1)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only for when there is no evaluation
        if args.evaluation_strategy == IntervalStrategy.NO and "loss" in logs:
            values = {"Training Loss": logs["loss"]}
            # First column is necessarily Step sine we're not in epoch eval strategy
            values["Step"] = state.global_step
            self.training_tracker.write_line(values)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.training_tracker is not None:
            values = {"Training Loss": "No log"}
            for log in reversed(state.log_history):
                if "loss" in log:
                    values["Training Loss"] = log["loss"]
                    break

            if self.first_column == "Epoch":
                values["Epoch"] = int(state.epoch)
            else:
                values["Step"] = state.global_step
            values["Validation Loss"] = metrics["eval_loss"]
            _ = metrics.pop("total_flos", None)
            _ = metrics.pop("epoch", None)
            _ = metrics.pop("eval_runtime", None)
            _ = metrics.pop("eval_samples_per_second", None)
            for k, v in metrics.items():
                if k == "eval_loss":
                    values["Validation Loss"] = v
                else:
                    splits = k.split("_")
                    name = " ".join([part.capitalize() for part in splits[1:]])
                    values[name] = v
            self.training_tracker.write_line(values)
            self.training_tracker.remove_child()
            self.prediction_bar = None
            # Evaluation takes a long time so we should force the next update.
            self._force_next_update = True

    def on_train_end(self, args, state, control, **kwargs):
        self.training_tracker.update(
            state.global_step, comment=f"Epoch {int(state.epoch)}/{state.num_train_epochs}", force_update=True
        )
        self.training_tracker = None
