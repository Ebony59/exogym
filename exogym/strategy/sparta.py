import math
import time
import wandb
import torch
import torch.distributed as dist

from typing import Optional, Union

from .communicate_optimize_strategy import (
    CommunicateOptimizeStrategy,
    CommunicationModule,
)
from .optim import OptimSpec
from .communicate import all_reduce, broadcast

def robust_stats(x):
    if x is None or (hasattr(x, "numel") and x.numel() == 0):
        return(None, None)
    mean = x.mean().item()
    k = max(1, int(0.95 * x.numel()))
    p95 = x.float().kthvalue(k).values.item()
    return (mean, p95)

class SparseCommunicator(CommunicationModule):
    """
    Communication module for sparse parameter communication (like SPARTA).
    """

    def __init__(self, index_selector, **kwargs):
        super().__init__(**kwargs)
        self.index_selector = index_selector
        self.iteration = 0
        self.cum_bytes = 0

    def communicate(self, model, rank: int, num_nodes: int, local_step: int) -> None:
        """Perform sparse communication."""
        if num_nodes > 1:
            t0 = time.perf_counter()
            total_selected = 0
            total_numel = 0
            bytes_this_step = 0
            
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue

                    self.index_selector.ensure_shadow(param)
                    indices_mask = self.index_selector.get_indices(
                        param, self.iteration
                    )

                    # Broadcasting a mask might be needed
                    broadcast(indices_mask, src=0)

                    sel = indices_mask.sum().item()
                    if sel == 0:
                        continue
                    total_selected += sel
                    total_numel += param.numel()
                    payload_bytes = sel * param.element_size()
                    bytes_this_step += payload_bytes
                    
                    sparse_data = param.data[indices_mask]
                    all_reduce(sparse_data, op=dist.ReduceOp.SUM)
                    sparse_data /= num_nodes

                    param.masked_scatter_(indices_mask, sparse_data)

                    if hasattr(self.index_selector, "post_apply"):
                        self.index_selector.post_apply(
                            param=param,
                            iteration=self.iteration,
                            optimizer=(self._get_optimizer() if hasattr(self, "strategy") else None),
                            mask=indices_mask,
                            reduced_values=sparse_data,
                        )

        self.iteration += 1
        self.cum_bytes += bytes_this_step

        if rank == 0:
            eff_p = (total_selected / total_numel) if total_numel else 0.0
            n = num_nodes
            ring_bytes = (2.0 * (n-1)/n) * bytes_this_step
            wandb.log({
                "selected_frac": eff_p,
                "payload_bytes": bytes_this_step,
                "ring_bytes_est": ring_bytes,
                "cum_payload_bytes": self.cum_bytes,
                "step_time_ms": (time.perf_counter() - t0) * 1e3
            }, step=local_step)

            opt = self._get_optimizer()
            rms_mean = rms_p95 = drift_mean = drift_p95 = None
            sample_count = 0

            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                n = p.numel()
                if n == 0:
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
                }, step=local_step)


    def _init_node(self, model, rank, num_nodes):
        pass

    def _get_optimizer(self):
        if not hasattr(self, "strategy"):
            return None
        for key in ("optimizer", "optim", "inner_optimizer", "outer_optimizer", "opt"):
            opt = getattr(self.strategy, key, None)
            if opt is not None:
                return opt
        return None


class SPARTAStrategy(CommunicateOptimizeStrategy):
    def __init__(
        self,
        optim_spec: Optional[Union[str, OptimSpec]] = None,
        p_sparta=0.005,
        **kwargs,
    ):

        # Create index selector and sparse communicator
        index_selector = RandomIndexSelector(p_sparta, enable_shadow_logging=True)
        sparse_comm = SparseCommunicator(index_selector)

        super().__init__(
            optim_spec=optim_spec, communication_modules=[sparse_comm], **kwargs
        )

        self.index_selector = index_selector


class IndexSelector:
    def __init__(self, p, enable_shadow_logging: bool = False):
        self.state = {}
        self.p = p
        self.enable_shadow_logging = enable_shadow_logging
        self._shadow = {}

    @torch.no_grad()
    def ensure_shadow(self, param):
        if not self.enable_shadow_logging:
            return None
        key = id(param)
        flat = param.data.view(-1)
        sh = self._shadow.get(key, None)
        if (
            sh is None
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
    # Update signature to match base class
    def get_indices(self, param, iteration):
        return torch.bernoulli(
            torch.full(param.shape, self.p, device=param.device)
        ).bool()


class ShuffledSequentialIndexSelector(IndexSelector):
    def __init__(self, p):
        # No model-dependent init here
        super().__init__(p)
        # Remove self.shuffled_state and self.index

    # Update signature to match base class
    def get_indices(self, param, iteration):
        num_total = param.numel()
        if num_total == 0:
            return torch.zeros_like(param, dtype=torch.bool)

        # Initialize state for this parameter if not seen before
        if param not in self.state:
            num_partitions = max(
                1, math.ceil(1.0 / self.p)
            )  # Ensure at least 1 partition
            shuffled_indices = torch.randperm(num_total, device=param.device)
            self.state[param] = {
                "num_partitions": num_partitions,
                "shuffled_indices": shuffled_indices,
            }

        param_state = self.state[param]
        num_partitions = param_state["num_partitions"]
        shuffled_indices = param_state["shuffled_indices"]

        # Determine the current chunk based on the iteration number
        current_chunk = iteration % num_partitions

        # Calculate chunk size and remainder for potentially uneven distribution
        chunk_size = num_total // num_partitions
        remainder = num_total % num_partitions

        # Calculate start and end indices for the current chunk
        start_index = current_chunk * chunk_size + min(current_chunk, remainder)
        # The end index calculation ensures the chunk size is correct, adding 1 for chunks getting the remainder
        end_index = start_index + chunk_size + (1 if current_chunk < remainder else 0)

        # Get the flat indices for the current chunk
        selected_flat_indices = shuffled_indices[start_index:end_index]

        # Create and return the boolean mask
        mask = torch.zeros(num_total, dtype=torch.bool, device=param.device)
        if (
            selected_flat_indices.numel() > 0
        ):  # Handle empty selection if num_total is very small
            mask[selected_flat_indices] = True
        return mask.view(param.shape)


class PartitionedIndexSelector(IndexSelector):
    def __init__(self, p):
        super().__init__(p)
        # Note: This class implicitly uses a step counter per parameter via self.state[param]["curr_partition"]
        # It doesn't need the global iteration number passed in.
        # To be consistent, we should update its signature, but the iteration argument would be unused.

    def _set_partition(self, param):
        param_state = self.state[param]
        param_state["curr_partition"] = 0
        # Ensure at least 1 partition
        num_partitions = max(1, min(math.ceil(1.0 / self.p), param.numel()))
        param_state["num_partitions"] = num_partitions
        if param.numel() > 0:
            param_state["partitions"] = (
                torch.rand(param.numel(), device=param.device).argsort()
                % num_partitions
            )
        else:
            # Handle zero-element tensors
            param_state["partitions"] = torch.empty(
                0, dtype=torch.long, device=param.device
            )

    # Update signature, though iteration is unused here
    def get_indices(self, param, iteration):
        # Handle zero-element tensors gracefully
        if param.numel() == 0:
            return torch.zeros_like(param, dtype=torch.bool)

        if param not in self.state:
            self.state[param] = {}
            self._set_partition(param)
        # Check if cycle needs reset BEFORE accessing partitions
        elif self.state[param]["curr_partition"] >= self.state[param]["num_partitions"]:
            self._set_partition(param)

        param_state = self.state[param]

        # Need to handle case where num_partitions might be 0 if numel was 0 during _set_partition
        # Although we added checks for numel=0, ensure partition access is safe
        if param_state["num_partitions"] == 0:
            return torch.zeros_like(
                param, dtype=torch.bool
            )  # Should not happen if numel > 0

        # Indices calculation requires reshaping the flat partitions result
        partition_indices = param_state["partitions"] == param_state["curr_partition"]
        indices_mask = partition_indices.view(
            param.shape
        ).bool()  # Reshape flat bool tensor to param shape

        param_state["curr_partition"] += 1

        return indices_mask
