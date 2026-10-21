"""All metric names are constants: `<split>/<name>` series, `best/...`, `final/...` summaries."""

FINAL_LOG_REGRET = "final/log10_regret"  # + "/<arm>": mean over tasks of log10(regret + eps)
TIME_PREFIX = "time/"  # + "<arm>_s": wall seconds per arm or controller
TIME_TASKS_S = "time/tasks_s"

# the template's example pipeline (tests/template)
TRAIN_MSE = "train/mse"
FINAL_TRAIN_MSE = "final/train_mse"
FINAL_EVAL_MSE = "final/eval_mse"
FINAL_W_ERROR = "final/w_error"

# the template's example pipeline (tests/template)
TRAIN_MSE = "train/mse"
FINAL_TRAIN_MSE = "final/train_mse"
FINAL_EVAL_MSE = "final/eval_mse"
FINAL_W_ERROR = "final/w_error"

# ei_readout
TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"
FINAL_VAL_LOSS = "final/val_loss"
EI_RATIO_MODE_MEDIAN = "final/ei_ratio_mode_median"
EI_RATIO_MODE_P10 = "final/ei_ratio_mode_p10"
EI_RATIO_SAMPLE_MEDIAN = "final/ei_ratio_sample_median"
EI_RATIO_SAMPLE_P10 = "final/ei_ratio_sample_p10"
