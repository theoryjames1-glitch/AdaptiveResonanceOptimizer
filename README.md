# AdaptiveResonanceOptimizer

```python
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class ARTDynamicsOptimizer(torch.optim.Optimizer):
    def __init__(self, params, hidden_dim, lr=5e-5, eta=0.05, lam=0.01, device="cpu"):
        defaults = dict(lr=lr, eta=eta, lam=lam)
        super().__init__(params, defaults)

        # single ART weight vector (prototype) for now
        self.w = torch.rand(hidden_dim, device=device)

    @torch.no_grad()
    def step(self, closure=None, hidden=None):
        """SGD + ART differential updates"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eta = group["eta"]

            # --- standard SGD update for GPT-2 weights ---
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.add_(-lr, p.grad)

            # --- ART update for memory trace w ---
            if hidden is not None:
                self.w += eta * (torch.min(hidden.mean(0), self.w) - self.w)

        return loss

    def art_loss(self, hidden):
        """Resonance penalty: pull h toward ART weight w"""
        return ((hidden - self.w) ** 2).mean()

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", output_hidden_states=True).to(device)

hidden_dim = model.config.hidden_size
optimizer = ARTDynamicsOptimizer(
    list(model.parameters()), hidden_dim=hidden_dim, lr=5e-5, eta=0.05, lam=0.01, device=device
)

prompts = [
    {"prompt": "Translate 'bonjour' to English:", "target": " hello"},
    {"prompt": "2 + 2 =", "target": " 4"},
    {"prompt": "The capital of France is", "target": " Paris"},
    {"prompt": "Hi, how are you?", "target": " I'm fine."},
]

def make_input_and_labels(prompt, target):
    full = tokenizer(prompt + target, return_tensors="pt").to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    labels = full.input_ids.clone()
    labels[:, : prompt_ids.size(1)] = -100
    return full, labels

for step, batch in enumerate(prompts * 2):
    inputs, labels = make_input_and_labels(batch["prompt"], batch["target"])
    outputs = model(**inputs, labels=labels)

    ce_loss = outputs.loss
    hidden = outputs.hidden_states[-1][:, -1, :]  # last token embedding

    # total loss = CE + ART resonance
    art_loss = optimizer.art_loss(hidden)
    total_loss = ce_loss + optimizer.param_groups[0]["lam"] * art_loss

    total_loss.backward()
    optimizer.step(hidden=hidden)
    optimizer.zero_grad()

    print(f"Step {step} | CE={ce_loss.item():.4f} | ART={art_loss.item():.4f}")
```
