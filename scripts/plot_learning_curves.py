import json, math, os
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

def load_tensorboard_data(log_dir):
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()

    eval_loss = ea.Scalars('eval/loss')

    steps = [x.step for x in eval_loss]
    losses = [x.value for x in eval_loss]
    ppls = [math.exp(loss) for loss in losses]

    return steps, losses, ppls

def plot_comparison():
    exp_steps, exp_loss, exp_ppl = load_tensorboard_data(
        "logs/explicit_gpt2/logs"
    )
    imp_steps, imp_loss, imp_ppl = load_tensorboard_data(
        "logs/implicit_gpt2/logs"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(exp_steps, exp_loss, 'b-', label='Explicit', linewidth=2)
    axes[0].plot(imp_steps, imp_loss, 'r-', label='Implicit', linewidth=2)
    axes[0].set_xlabel('Trainig Steps', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('E1: Learning Curves (Loss)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(exp_steps, exp_ppl, 'b-', label='Explicit', linewidth=2)
    axes[1].plot(imp_steps, imp_ppl, 'r-', label='Implicit', linewidth=2)
    axes[1].set_xlabel('Trainig Steps', fontsize=12)
    axes[1].set_ylabel('Perplexity', fontsize=12)
    axes[1].set_title('E1: Learning Curves (Perplexity)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/E1_learning_curves.png', dpi=300)
    print("Saved: results/E1_learning_curves.png")

if __name__ == "__main__":
    plot_comparison()
