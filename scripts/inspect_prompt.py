import sys
from pathlib import Path
import torch

# Add root folder to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt_optimizer.training.config import TrainingConfig
from prompt_optimizer.model.optimizer_model import PromptOptimizerModel

def inspect(checkpoint_path: str, prompt: str, task: str):
    print(f"\n=== Inspecting checkpoint: {checkpoint_path} ===")
    cfg = TrainingConfig()
    model = PromptOptimizerModel(cfg)
    try:
        model.load(checkpoint_path)
        model.eval()
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return

    device = next(model.parameters()).device
    encoded = model.token_scorer.tokenize([prompt], device=device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    task_indices = torch.tensor([cfg.task_to_idx(task)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits = model.token_scorer(input_ids, attention_mask, task_indices)
        probs = torch.sigmoid(logits / model.gumbel_selector.tau_end)
        
    tokens = model.token_scorer.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    
    print(f"{'Token':<15} | {'Logit':<10} | {'Probability':<12} | {'Kept (thresh=0.5)':<10}")
    print("-" * 55)
    for tok, logit, prob in zip(tokens, logits[0].tolist(), probs[0].tolist()):
        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        kept = "YES" if prob >= 0.5 else "NO"
        print(f"{tok:<15} | {logit:+.4f} | {prob:.4f} | {kept}")

if __name__ == "__main__":
    prompts = [
        ("please write me code for binary search", "code"),
        ("Can you write a Python function that implements binary search on a sorted list?", "code")
    ]
    checkpoints = [
        "checkpoints/final.pt"
    ]
    for cp in checkpoints:
        if Path(cp).exists():
            for pr, task in prompts:
                print(f"\nPrompt: '{pr}' (task: {task})")
                inspect(cp, pr, task)
