from abc import abstractmethod
from typing import Optional
import time
import wandb
import torch
import argparse

from exogym.strategy.communicate import broadcast, all_reduce, all_gather
from exogym.strategy.strategy import SimpleReduceStrategy, Strategy
from exogym.trainer import Trainer
from exogym.strategy.optim import OptimSpec
from exogym.strategy.sparta import RandomIndexSelector
from exogym.aux.utils import get_device

from nanogpt import GPT, GPTConfig, get_dataset

NUM_NODES = 4

### PLAYGROUND
### This is a minimal configuration for training a nanogpt model with a given strategy.
### The strategy can be swapped out for custom logic by writing a new strategy class.

def robust_stats(x: torch.Tensor):
    if x is None or x.numel() == 0:
        return (None, None)
    mean = x.mean().item()
    k = max(1, int(0.95 * x.numel()))
    p95 = x.float().kthvalue(k).values.item()
    return (mean, p95)


# class IndexSelector:
#     def __init__(self, p):
#         self.state = {}
#         self.p = p

#     @abstractmethod
#     def get_indices(self, param, iteration):
#         ...

class IndexSelector:
    def __init__(self, p, enable_shadow_logging: bool = False):
        self.state = {}
        self.p = p
        self.enable_shadow_logging = enable_shadow_logging
        self._shadow = {}
        self._optimizer = None  # <-- add

    def set_optimizer(self, optim):
        self._optimizer = optim

    @torch.no_grad()
    def ensure_shadow(self, param):
        if not self.enable_shadow_logging:
            return None
        key = id(param)
        flat = param.data.view(-1)
        sh = self._shadow.get(key, None)
        if (sh is None
            or sh.numel() != flat.numel()
            or sh.device != flat.device
            or sh.dtype != flat.dtype
        ):
            self._shadow[key] = flat.detach().clone()
        return self._shadow[key]

    def get_shadow_vector(self, param):
        return self._shadow.get(id(param), None)
    
    # Add iteration argument to the base class signature
    def get_indices(self, param, iteration):
        # Default implementation returns all indices (mask of Trues)
        self.ensure_shadow(param)
        return torch.ones_like(param, dtype=torch.bool)

    @torch.no_grad()
    def post_apply(self, *, param, iteration, optimizer=None, mask=None, reduced_values=None):
        if not self.enable_shadow_logging:
            return
        if mask is None or reduced_values is None:
            return
        sh = self.ensure_shadow(param)
        if sh is None:
            return
        sh[mask.view(-1)] = reduced_values.detach().to(sh.dtype)


class RandomIndexSelector(IndexSelector):
    def get_indices(self, param, iteration):
        return torch.bernoulli(
            torch.full(param.shape, self.p, device=param.device)
        ).bool()


class AdamSecondMomentSelector(IndexSelector):
    def __init__(self,
            p,
            alpha=0.5,
            eps=1e-12,
            p_floor=1e-6,
            p_ceil=0.2,
            mix_uniform=0.1,
            enable_shadow_logging: bool = False
        ):
        super().__init__(p, enable_shadow_logging=enable_shadow_logging)
        self._shadow = {}
        self.alpha = alpha
        self.eps = eps
        self.p_floor = p_floor
        self.p_ceil = p_ceil
        self.mix_uniform = 0.1

    @torch.no_grad()
    def get_indices(self, param, iteration):
        n = param.numel()
        if n == 0:
            return torch.zeros_like(param, dtype=torch.bool)

        opt = getattr(self, "_optimizer", None)
        st = (opt.state.get(param, None) if opt is not None else None)
        v = (st.get("exp_avg_sq", None) if st is not None else None)
        if v is None:
            self.ensure_shadow(param)
            return torch.bernoulli(
                torch.full(param.shape, self.p, device=param.device)
            ).bool()

        # importance scores: RMS^alpha (alpha in (0,1] flattens peaky distributions)
        w = (v.detach().view(-1).sqrt() + self.eps).pow(self.alpha)

        # scale to expected k = ceil(p * n)
        k = max(1, int(self.p * n))
        scale_factor = (k / (w.sum() + self.eps)).item()
        p_adam = (scale_factor * w).clamp_(self.p_floor, self.p_ceil)

        p_uni = (float(k) / n)
        p_final = (1.0 - self.mix_uniform) * p_adam + self.mix_uniform * p_uni

        # Bernoulli draws
        mask = torch.bernoulli(p_final).bool().view_as(param)

        self.ensure_shadow(param)
        return mask


class SPARTAStrategy(Strategy):
    def __init__(
        self,
        optim_spec: Optional[str | OptimSpec] = None,
        p_sparta=0.005,
        **kwargs,
    ):

        index_selector = AdamSecondMomentSelector(
            p_sparta,
            mix_uniform=0.1,
            enable_shadow_logging=True
        )

        super().__init__(**kwargs)
        
        self.optim_spec = optim_spec if isinstance(optim_spec, OptimSpec) else OptimSpec.from_str(optim_spec)
        self.index_selector = index_selector
        self._cum_bytes = 0

    def step(self, ):
        current_step = self.local_step
        
        t0 = time.perf_counter()
        total_selected = 0
        total_numel = 0
        bytes_this_step = 0

        if getattr(self.index_selector, "_optimizer", None) is None and hasattr(self, "optim"):
            self.index_selector.set_optimizer(self.optim)
        
        with torch.no_grad():
            for param in self.model.parameters():
                if not param.requires_grad or param.grad is None:
                    continue

                self.index_selector.ensure_shadow(param)

                indices_mask = self.index_selector.get_indices(
                    param, self.local_step
                )

                broadcast(indices_mask, src=0)

                sel = int(indices_mask.sum().item())
                if sel == 0:
                    continue

                total_selected += sel
                total_numel += param.numel()
                bytes_this_step += sel * param.element_size()
                
                sparse_data = param.data[indices_mask]
                
                all_reduce(sparse_data, op=torch.distributed.ReduceOp.SUM)
                sparse_data /= self.num_nodes

                param.masked_scatter_(indices_mask, sparse_data)
                self.index_selector.post_apply(
                    param=param,
                    iteration=self.local_step,
                    optimizer=self.optim,
                    mask=indices_mask,
                    reduced_values=sparse_data,
                )

        self.optim.step()
        super().step()

        # logging at rank 0
        self._cum_bytes += bytes_this_step
        if getattr(self, "rank", 0) == 0:
            eff_p = (total_selected / total_numel) if total_numel else 0.0
            n = self.num_nodes
            ring_bytes = (2.0 * (n-1)/n) * bytes_this_step
            wandb.log({
                "selected_frac": eff_p,
                "payload_bytes": bytes_this_step,
                "ring_bytes_est": ring_bytes,
                "cum_payload_bytes": self._cum_bytes,
                "step_time_ms": (time.perf_counter() - t0) * 1e3
            }, step=current_step)

            rms_mean = rms_p95 = drift_mean = drift_p95 = None
            sample_count = 0
            opt = getattr(self, "optim", None)
    
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.numel() == 0:
                    continue
    
                # Adam second moment (RMS)
                rms_layer = None
                if opt is not None and p in opt.state and "exp_avg_sq" in opt.state[p]:
                    v = opt.state[p]["exp_avg_sq"].detach().view(-1)
                    k = min(4096, v.numel())
                    idx = torch.randint(0, v.numel(), (k,), device=v.device)
                    rms_vals = (v[idx].sqrt()).float()
                    rms_layer = rms_vals
    
                # Drift
                drift_layer = None
                shadow = self.index_selector.get_shadow_vector(p)
                if torch.is_tensor(shadow) and shadow.shape[0] == p.data.numel():
                    theta = p.data.view(-1)
                    k = min(4096, theta.numel())
                    idx = torch.randint(0, theta.numel(), (k,), device=theta.device)
                    drift_layer = (theta[idx] - shadow[idx]).abs().float()
    
                if rms_layer is not None:
                    m, p95 = robust_stats(rms_layer)
                    rms_mean = (m if rms_mean is None else (rms_mean + m))
                    rms_p95 = (p95 if rms_p95 is None else (rms_p95 + p95))
                if drift_layer is not None:
                    m, p95 = robust_stats(drift_layer)
                    drift_mean = (m if drift_mean is None else (drift_mean + m))
                    drift_p95  = (p95 if drift_p95 is None else (drift_p95 + p95))
    
                sample_count += 1
            
            if sample_count > 0:
                def avg_or_none(x):
                    return (x / sample_count) if isinstance(x, (int,float)) else (x / sample_count if x is not None else None)
    
                wandb.log({
                    "rms_grad_mean": avg_or_none(rms_mean),
                    "rms_grad_p95":  avg_or_none(rms_p95),
                    "drift_mean":    avg_or_none(drift_mean),
                    "drift_p95":     avg_or_none(drift_p95),
                }, step=current_step)
        
        
def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="owt")
    args = arg_parser.parse_args()
    dataset = args.dataset

    # Get datasets - this will take a while the first time, as the dataset has to be imported and processed.
    train_dataset, vocab_size = get_dataset(
        dataset,
        block_size=1024,
        device="cpu",
        start_pc=0.0,
        end_pc=0.005 * NUM_NODES if dataset == "owt" else 0.99,
    )
    val_dataset, vocab_size = get_dataset(
        dataset, 
        block_size=1024, 
        device="cpu",
        start_pc=0.99, 
        end_pc=1.0
    )

    device = get_device()

    gpt_config = GPTConfig.gpt2_sbase()
    gpt_config.vocab_size = vocab_size
    model = GPT(gpt_config)

    # Create trainer
    trainer = Trainer(
        model,
        train_dataset,
        val_dataset,
    )

    ## STRATEGY - This is where we define custom logic

    # to default back to data parallel training:
    # strategy = SimpleReduceStrategy(
    #     optim_spec=OptimSpec(torch.optim.AdamW, lr=0.0004),
    #     lr_scheduler="lambda_cosine",
    #     lr_scheduler_kwargs={
    #         "warmup_steps": 1000,
    #         "cosine_anneal": True,
    #     },
    #     max_norm=1.0,
    # )

    strategy = SPARTAStrategy(
        optim_spec=OptimSpec(torch.optim.AdamW, lr=0.0004),
        lr_scheduler="lambda_cosine",
        lr_scheduler_kwargs={
            "warmup_steps": 1000,
            "cosine_anneal": True,
        },
        max_norm=1.0,
        p=0.005,
    )

    # Train it!
    trainer.fit(
        num_epochs=1,
        max_steps=5000,
        strategy=strategy,
        num_nodes=NUM_NODES,
        device=device,
        batch_size=16,
        minibatch_size=8, # Gradient accumulation to ensure we can fit in memory. Make this even lower for smaller devices. 
        shuffle=False,
        val_size=256,
        val_interval=100,
        wandb_project="exo-sparta",
        run_name="sparta-adam-mix0.1",
    )


if __name__ == "__main__":
    main()
