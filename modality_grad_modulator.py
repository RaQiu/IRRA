"""
modality_grad_modulator.py
==========================
Cross-Modal Adaptive Gradient Modulation Plugin

A model-agnostic plugin that adaptively adjusts gradients based on the modal
specificity of neurons and the relative learning progress of each modality.
Supports single-GPU and multi-GPU (DDP) training.

Extracted from a modified IRRA implementation with zero behavioral difference.

Usage (with IRRA as example):
    See IRRA_INTEGRATION_EXAMPLE at the bottom of this file.
"""

import logging
import os
import pickle
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False

logger = logging.getLogger("ModalityGradModulator")


# ======================================================================
# Helpers
# ======================================================================

def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DDP / FSDP wrapped model."""
    return model.module if hasattr(model, 'module') else model


# Type alias for module filter functions
ModuleFilter = Callable[[str, nn.Module], bool]


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class ModulationConfig:
    """All hyperparameters for gradient modulation.

    Defaults match the modified-IRRA code so that using this config
    directly reproduces the original behaviour.
    """
    # -- Modal specificity --
    tau: int = 1                # Modal specificity check interval (steps)

    # -- Loss-based scoring --
    gammb: float = 1.65         # Loss-difference exponent (Eqn. score amplification)

    # -- Sigmoid stability modulation (Eqn.9) --
    gaeta: float = 1.0          # Activation threshold (half-epoch rounds) & oscillation weight
    alpha: float = 0.9          # Sigmoid steepness
    beta: float = 0.999         # Training progress decay rate

    # -- Feature flags --
    enabled: bool = True        # Master switch for gradient modulation
    fig1c_enabled: bool = True  # Whether to collect Fig1(c) gradient noise data
    fig1c_noise_std: float = 0.1

    # -- Output --
    output_dir: str = "./logs"


# ======================================================================
# Main Plugin
# ======================================================================

class ModalityGradModulator:
    """
    Model-agnostic cross-modal gradient modulation plugin.

    Lifecycle (per training run)::

        modulator = ModalityGradModulator(config)
        modulator.attach(model, shared_filter, img_enc_filter, txt_enc_filter)

        for epoch in range(num_epochs):
            modulator.on_epoch_start(model)

            for step, batch in enumerate(loader):
                optimizer.zero_grad()

                modulator.pre_forward(model)
                ret = model(batch)
                modulator.capture('normal')

                with torch.no_grad():
                    e_txt_ret = erase_forward(model, batch, 'txt')
                    modulator.capture('e_txt')
                    e_img_ret = erase_forward(model, batch, 'img')
                    modulator.capture('e_img')

                loss.backward()

                modulator.post_backward(
                    model, step, len(loader), batch_size,
                    all_loss, e_txt_loss, e_img_loss,
                )

                optimizer.step()

            stats = modulator.on_epoch_end(model, epoch)
    """

    def __init__(self, config: ModulationConfig):
        self.config = config

        # ---- Module registries (populated by attach) ----
        self._shared_modules: Dict[str, nn.Module] = {}
        self._img_enc_modules: Dict[str, nn.Module] = {}
        self._txt_enc_modules: Dict[str, nn.Module] = {}
        self._module_to_name: Dict[int, str] = {}   # id(module) -> name
        self._module_by_id: Dict[int, nn.Module] = {}  # id(module) -> module

        # ---- Hook state ----
        self._activations: Dict[int, torch.Tensor] = {}
        self._handles: List = []
        self._captured: Dict[str, Dict[int, torch.Tensor]] = {}

        # ---- Modal specificity counts (key = module name) ----
        self._modal_img_counts: Dict[str, torch.Tensor] = {}
        self._modal_txt_counts: Dict[str, torch.Tensor] = {}
        self._modal_img_tot: Dict[str, torch.Tensor] = {}
        self._modal_txt_tot: Dict[str, torch.Tensor] = {}
        self._diff: Dict[str, torch.Tensor] = {}
        self._last: Dict[str, torch.Tensor] = {}

        # ---- Loss accumulators ----
        self._delta_img_loss: float = 0.0
        self._delta_txt_loss: float = 0.0
        self._tot: int = 0
        self._times: int = 0

        # ---- Gradient ratio tracking ----
        self._grad_ratio: Dict[str, List] = {}
        self._grad_ratio_stats: Dict[str, List[float]] = {
            "mean": [], "max": [], "min": [],
        }

        # ---- Fig1(c) state ----
        self._fig1c_data: Dict[str, list] = {
            "epoch_list": [],
            "text_no_noise": [],
            "text_with_noise": [],
            "img_no_noise": [],
            "img_with_noise": [],
        }
        self._grad_cache: Dict[str, List[float]] = {
            "text_no_noise": [],
            "text_with_noise": [],
            "img_no_noise": [],
            "img_with_noise": [],
        }
        self._noise_experiment_done: bool = False

    # ==================================================================
    # Public API
    # ==================================================================

    def attach(
        self,
        model: nn.Module,
        shared_filter: ModuleFilter,
        img_enc_filter: ModuleFilter,
        txt_enc_filter: ModuleFilter,
    ):
        """Register module categories.

        Args:
            model: The multimodal model (raw or DDP-wrapped).
            shared_filter: ``(name, module) -> bool`` for modules that receive
                neuron-level gradient modulation (typically cross-modal layers).
            img_enc_filter: ``(name, module) -> bool`` for image-encoder leaves.
            txt_enc_filter: ``(name, module) -> bool`` for text-encoder leaves.
        """
        base = unwrap_model(model)
        self._shared_modules.clear()
        self._img_enc_modules.clear()
        self._txt_enc_modules.clear()
        self._module_to_name.clear()
        self._module_by_id.clear()

        for name, module in base.named_modules():
            is_leaf = not list(module.children())
            has_params = bool(list(module.parameters()))
            if not (is_leaf and has_params):
                continue
            mid = id(module)
            if shared_filter(name, module):
                self._shared_modules[name] = module
                self._module_to_name[mid] = name
                self._module_by_id[mid] = module
            if img_enc_filter(name, module):
                self._img_enc_modules[name] = module
                self._module_to_name[mid] = name
            if txt_enc_filter(name, module):
                self._txt_enc_modules[name] = module
                self._module_to_name[mid] = name

        logger.info(
            "Attached: %d shared, %d img_enc, %d txt_enc modules",
            len(self._shared_modules),
            len(self._img_enc_modules),
            len(self._txt_enc_modules),
        )

    def on_epoch_start(self, model: nn.Module):
        """Call at the start of each training epoch."""
        self._noise_experiment_done = False

    def pre_forward(self, model: nn.Module):
        """Register forward hooks.  Call ONCE before the first of the three
        forward passes (normal / e_txt / e_img)."""
        self._handles = []
        for module in self._shared_modules.values():
            self._handles.append(module.register_forward_hook(self._hook_fn))

    def capture(self, key: str):
        """Snapshot current activations under *key* and clear the buffer.

        Call after each forward pass::

            capture('normal')   # after model(batch)
            capture('e_txt')    # after erase-text forward
            capture('e_img')    # after erase-image forward
        """
        self._captured[key] = dict(self._activations)
        self._activations.clear()

    def post_backward(
        self,
        model: nn.Module,
        step: int,
        data_len: int,
        batch_size: int,
        all_loss,
        e_txt_loss,
        e_img_loss,
    ):
        """Perform gradient modulation.  Call AFTER ``loss.backward()``,
        BEFORE ``optimizer.step()``.

        Args:
            model: The model (raw or DDP-wrapped).
            step: Current step within the epoch (0-based).
            data_len: Total steps per epoch (``len(train_loader)``).
            batch_size: Batch size for the current step.
            all_loss: Loss from the normal forward pass (tensor or float).
            e_txt_loss: Loss from the text-erased forward pass.
            e_img_loss: Loss from the image-erased forward pass.
        """
        # Fig1(c) noise experiment (once per epoch, at the first step)
        # Runs regardless of whether modulation is enabled, matching original behavior
        if self.config.fig1c_enabled and not self._noise_experiment_done:
            self._fig1c_noise_experiment(model)
            self._noise_experiment_done = True

        if not self.config.enabled:
            self._remove_hooks()
            self._captured.clear()
            return

        # Core gradient modulation
        all_act = self._captured.get('normal', {})
        e_txt_act = self._captured.get('e_txt', {})
        e_img_act = self._captured.get('e_img', {})

        self._modulate(
            model, step, data_len, batch_size,
            all_act, e_txt_act, e_img_act,
            all_loss, e_txt_loss, e_img_loss,
        )

        # Cleanup
        self._captured.clear()

    def on_epoch_end(self, model: nn.Module, epoch: int) -> dict:
        """Finalize epoch: save fig1c data, log statistics, reset counters.

        Returns:
            dict with ``cnt_txt``, ``cnt_img``, ``rho_num``,
            ``grad_ratio_mean`` (if available).
        """
        stats: dict = {}

        # Save Fig1(c) data
        if self.config.fig1c_enabled:
            self._save_fig1c_data(epoch)

        # Log per-module and global modal neuron counts
        cnt_txt, cnt_img, equal = 0, 0, 0
        for name in list(self._modal_txt_counts.keys()):
            if name not in self._modal_img_counts:
                continue
            mtxt = (self._modal_txt_counts[name] > self._modal_img_counts[name]).sum().item()
            mimg = (self._modal_img_counts[name] > self._modal_txt_counts[name]).sum().item()
            meq = (self._modal_img_counts[name] == self._modal_txt_counts[name]).sum().item()
            logger.info("%s : cnt_txt: %d, cnt_img: %d, equal: %d", name, mtxt, mimg, meq)
            cnt_txt += mtxt
            cnt_img += mimg
            equal += meq

        epsilon = 1e-8
        rho_num = cnt_img / (cnt_txt + epsilon)
        logger.info("[ALL] : cnt_txt: %d, cnt_img: %d, equal: %d", cnt_txt, cnt_img, equal)
        logger.info("[ALL] : rho_num: %.4f", rho_num)
        stats.update(cnt_txt=cnt_txt, cnt_img=cnt_img, rho_num=rho_num)

        # Gradient ratio statistics
        if self._grad_ratio_stats["mean"]:
            avg_mean = sum(self._grad_ratio_stats["mean"]) / len(self._grad_ratio_stats["mean"])
            avg_max = sum(self._grad_ratio_stats["max"]) / len(self._grad_ratio_stats["max"])
            avg_min = sum(self._grad_ratio_stats["min"]) / len(self._grad_ratio_stats["min"])
            logger.info(
                "[Epoch %d] Grad Ratio Stats - Mean: %.4f, Max: %.4f, Min: %.4f",
                epoch, avg_mean, avg_max, avg_min,
            )
            stats['grad_ratio_mean'] = avg_mean

        # Reset per-epoch state
        self._grad_ratio_stats = {"mean": [], "max": [], "min": []}
        self._modal_img_counts.clear()
        self._modal_txt_counts.clear()

        return stats

    # ==================================================================
    # Internal – Hook
    # ==================================================================

    def _hook_fn(self, module: nn.Module, input, output):
        """Forward hook: capture ``input[0]`` (first positional arg)."""
        self._activations[id(module)] = input[0].detach()

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ==================================================================
    # Internal – Core Modulation
    # ==================================================================

    def _modulate(
        self, model, step, data_len, batch_size,
        all_act, e_txt_act, e_img_act,
        all_loss, e_txt_loss, e_img_loss,
    ):
        """Core gradient modulation – mirrors original ``modulation()`` exactly."""
        base = unwrap_model(model)
        base.eval()

        cfg = self.config

        # Accumulate loss deltas
        _to_float = lambda v: v.item() if torch.is_tensor(v) else float(v)
        self._delta_img_loss += _to_float(e_img_loss) - _to_float(all_loss)
        self._delta_txt_loss += _to_float(e_txt_loss) - _to_float(all_loss)
        self._tot += 1

        with torch.no_grad():
            # ---- Per-step: modal specificity check ----
            if step % cfg.tau == 0:
                for mid, act in all_act.items():
                    if mid not in e_img_act or mid not in e_txt_act:
                        continue
                    name = self._module_to_name.get(mid)
                    if name is None:
                        continue

                    # Delta activations – handle (L, B, D) vs (B, L, D) layout
                    delta_img = torch.abs(act - e_img_act[mid])
                    if delta_img.size(0) != batch_size:
                        delta_img = delta_img.permute(1, 0, 2)
                    delta_img = torch.mean(torch.mean(delta_img, dim=1), dim=0)

                    delta_txt = torch.abs(act - e_txt_act[mid])
                    if delta_txt.size(0) != batch_size:
                        delta_txt = delta_txt.permute(1, 0, 2)
                    delta_txt = torch.mean(torch.mean(delta_txt, dim=1), dim=0)

                    txt_specificity = delta_txt - delta_img
                    img_specificity = delta_img - delta_txt
                    num_neurons = delta_img.size(-1)

                    indicate = torch.ones(num_neurons, device=act.device) * -1
                    indicate[img_specificity > 0] = 1
                    indicate[txt_specificity > 0] = 0
                    tied = (img_specificity == 0) & (txt_specificity == 0)
                    indicate[tied] = random.randint(0, 1)

                    if name not in self._modal_img_counts:
                        self._modal_img_counts[name] = torch.ones(num_neurons, device=act.device) * -1
                        self._modal_txt_counts[name] = torch.ones(num_neurons, device=act.device) * -1

                    self._modal_img_counts[name] += (indicate == 1)
                    self._modal_txt_counts[name] += (indicate == 0)

            # ---- Half-epoch: gradient modulation ----
            if (step + 1) % (data_len // 2) == 0:
                self._times += 1

                # >>> DDP: synchronize counts & loss deltas across GPUs <<<
                self._sync_distributed()

                # Lazy-init per-module state
                for mid, act in all_act.items():
                    name = self._module_to_name.get(mid)
                    if name is None:
                        continue
                    num_neurons = act.size(-1)
                    if name not in self._modal_img_tot:
                        self._modal_img_tot[name] = torch.zeros(num_neurons, device=act.device)
                        self._modal_txt_tot[name] = torch.zeros(num_neurons, device=act.device)
                    if name not in self._diff:
                        self._diff[name] = torch.zeros(num_neurons, device=act.device)

                # Score from loss deltas
                score = torch.tensor(
                    [self._delta_txt_loss / self._tot,
                     self._delta_img_loss / self._tot],
                    device="cuda",
                )
                ratio = F.softmax(score, dim=0)
                r_min, _ = torch.min(ratio, dim=0)
                iscore = (ratio - r_min) ** cfg.gammb
                iscore_txt = iscore[0].item()
                iscore_img = iscore[1].item()

                logger.info(
                    "Step %d, delta_txt_loss = %s , delta_img_loss = %s",
                    step, self._delta_txt_loss, self._delta_img_loss,
                )
                logger.info("iscore_txt = %s, iscore_img = %s", iscore_txt, iscore_img)

                # ---- Per-module pen computation ----
                for mid, act in all_act.items():
                    name = self._module_to_name.get(mid)
                    module = self._module_by_id.get(mid)
                    if module is None or name is None:
                        continue
                    if name not in self._modal_img_counts:
                        continue

                    num_neurons = act.size(-1)
                    pen = torch.zeros(num_neurons, device=act.device)
                    now = torch.zeros(num_neurons, device=act.device)

                    now += (self._modal_img_counts[name] > self._modal_txt_counts[name]).to(torch.int)

                    if name in self._last:
                        self._diff[name] = self._diff[name] + (now != self._last[name]).to(torch.int)

                    pen_img = (self._modal_img_counts[name] > self._modal_txt_counts[name])
                    pen += pen_img * iscore_img
                    pen_txt = (self._modal_img_counts[name] < self._modal_txt_counts[name])
                    pen += pen_txt * iscore_txt

                    # Sigmoid stability modulation (Eqn.9)
                    if self._times >= cfg.gaeta:
                        k = max(iscore_img, iscore_txt) / 10
                        n_tensor = torch.tensor(self._times, dtype=torch.float32, device="cuda")
                        R = cfg.gaeta * self._diff[name] / self._times
                        f_R = 2.0 / (1.0 + torch.exp(-cfg.alpha * (R - 1.0))) - 1.0
                        g_n = 1.0 / (1.0 + cfg.beta * n_tensor)
                        delta_z = k * f_R * g_n
                        pen = torch.where(pen != 0, pen + delta_z, pen)

                        print("DIFF", torch.max(self._diff[name]), torch.min(self._diff[name]))
                        print("Delta_z", torch.max(delta_z), torch.min(delta_z))
                        print("@Instruct Before", torch.max(pen), torch.min(pen))

                    pen = 1 - pen
                    print("@Instruct After", torch.max(pen), torch.min(pen))
                    pen = torch.clamp(pen, min=0, max=1)

                    # Apply pen to module gradients
                    # NOTE: The size() comparison deliberately matches the original
                    # code (``param.grad.size()[1] == pen.size()`` compares int to
                    # torch.Size, which is False for >1-D params).  This preserves
                    # zero-difference behaviour.
                    for param in module.parameters():
                        if param.grad is None:
                            continue
                        grad_before = param.grad.clone()
                        if len(param.grad.size()) > 1 and param.grad.size()[1] == pen.size():
                            param.grad *= pen.unsqueeze(0)
                        elif len(param.grad.size()) == 1 and param.grad.size() == pen.size():
                            param.grad *= pen

                        self._last[name] = now

                        # Gradient ratio tracking
                        grad_eps = 1e-8
                        grad_ratio = torch.where(
                            grad_before.abs() > grad_eps,
                            param.grad / (grad_before + grad_eps),
                            torch.ones_like(grad_before),
                        )
                        if name not in self._grad_ratio:
                            self._grad_ratio[name] = []
                        self._grad_ratio[name].append(grad_ratio)

                        ratio_mean = grad_ratio.mean().item()
                        ratio_max = grad_ratio.max().item()
                        ratio_min = grad_ratio.min().item()
                        self._grad_ratio_stats["mean"].append(ratio_mean)
                        self._grad_ratio_stats["max"].append(ratio_max)
                        self._grad_ratio_stats["min"].append(ratio_min)

                        logger.info(
                            "Module %s - Param shape: %s | "
                            "Grad ratio: mean=%.4f, max=%.4f, min=%.4f",
                            name, param.grad.shape,
                            ratio_mean, ratio_max, ratio_min,
                        )

                # ---- Encoder-level suppression ----
                for _name, module in self._img_enc_modules.items():
                    for param in module.parameters():
                        if param.grad is not None:
                            param.grad *= (1 - iscore_img)

                for _name, module in self._txt_enc_modules.items():
                    for param in module.parameters():
                        if param.grad is not None:
                            param.grad *= (1 - iscore_txt)

                # Reset loss accumulators
                self._delta_img_loss = 0.0
                self._delta_txt_loss = 0.0
                self._tot = 0

        # Cleanup
        self._remove_hooks()
        base.train()

    # ==================================================================
    # Internal – Multi-GPU Synchronization
    # ==================================================================

    def _sync_distributed(self):
        """All-reduce modal counts and loss deltas across GPUs.

        Called at each half-epoch modulation point so that every GPU computes
        an identical ``pen`` vector and modifies gradients identically,
        keeping DDP parameters in sync.
        """
        if not (_DIST_AVAILABLE and dist.is_initialized()):
            return

        world_size = dist.get_world_size()
        if world_size <= 1:
            return

        # Sync modal neuron counts
        for name in list(self._modal_img_counts.keys()):
            dist.all_reduce(self._modal_img_counts[name], op=dist.ReduceOp.SUM)
            dist.all_reduce(self._modal_txt_counts[name], op=dist.ReduceOp.SUM)

        # Sync DIFF (oscillation counts) – only tensors that exist
        for name in list(self._diff.keys()):
            dist.all_reduce(self._diff[name], op=dist.ReduceOp.SUM)

        # Sync loss deltas and tot (pack into a single tensor for efficiency)
        sync_buf = torch.tensor(
            [self._delta_txt_loss, self._delta_img_loss, float(self._tot)],
            device="cuda",
        )
        dist.all_reduce(sync_buf, op=dist.ReduceOp.SUM)
        self._delta_txt_loss = sync_buf[0].item()
        self._delta_img_loss = sync_buf[1].item()
        self._tot = int(sync_buf[2].item())

        logger.debug("Synced modulation state across %d GPUs", world_size)

    # ==================================================================
    # Internal – Fig1(c) Gradient Noise Analysis
    # ==================================================================

    def _fig1c_noise_experiment(self, model: nn.Module):
        """Run the gradient noise sensitivity experiment (once per epoch).

        Adds Gaussian noise to gradients, measures Frobenius norms, restores.
        """
        # 1. No-noise baselines
        self._collect_fig1c_gradient(model, is_noise=False, modality="text")
        self._collect_fig1c_gradient(model, is_noise=False, modality="img")

        # 2. Text modality with noise
        text_cache = self._add_gaussian_noise(model, "text")
        self._collect_fig1c_gradient(model, is_noise=True, modality="text")
        self._restore_gradient(model, "text", text_cache)

        # 3. Image modality with noise
        img_cache = self._add_gaussian_noise(model, "img")
        self._collect_fig1c_gradient(model, is_noise=True, modality="img")
        self._restore_gradient(model, "img", img_cache)

    def _collect_fig1c_gradient(self, model, is_noise: bool, modality: str) -> float:
        modules = self._img_enc_modules if modality == "img" else self._txt_enc_modules
        grad_norms = []
        for module in modules.values():
            for param in module.parameters():
                if param.grad is not None:
                    grad_norms.append(torch.norm(param.grad, p='fro').item())

        avg = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        cache_key = f"{modality}_{'with_noise' if is_noise else 'no_noise'}"
        self._grad_cache[cache_key].append(avg)
        logger.info("fig1c: %s", self._grad_cache[cache_key])
        return avg

    def _add_gaussian_noise(self, model, modality: str) -> dict:
        modules = self._img_enc_modules if modality == "img" else self._txt_enc_modules
        cache: Dict[str, Dict[str, torch.Tensor]] = {}
        noise_std = self.config.fig1c_noise_std
        for name, module in modules.items():
            cache[name] = {}
            for pname, param in module.named_parameters():
                if param.grad is not None:
                    cache[name][pname] = param.grad.clone()
                    param.grad += torch.randn_like(param.grad) * noise_std
        return cache

    def _restore_gradient(self, model, modality: str, cache: dict):
        modules = self._img_enc_modules if modality == "img" else self._txt_enc_modules
        for name, module in modules.items():
            if name not in cache:
                continue
            for pname, param in module.named_parameters():
                if pname in cache[name] and param.grad is not None:
                    param.grad = cache[name][pname]

    def _save_fig1c_data(self, epoch: int):
        # Only rank-0 saves
        if _DIST_AVAILABLE and dist.is_initialized() and dist.get_rank() != 0:
            return

        for key in self._grad_cache:
            vals = self._grad_cache[key]
            self._fig1c_data[key].append(
                sum(vals) / len(vals) if vals else 0.0
            )
        self._fig1c_data["epoch_list"].append(epoch)

        save_dir = os.path.join(self.config.output_dir, "fig1c_data")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "fig1c_gradient_data.pkl")

        with open(save_path, 'wb') as f:
            pickle.dump(self._fig1c_data, f)

        for key in self._grad_cache:
            self._grad_cache[key] = []

        logger.info("[Fig1(c)] Epoch %d gradient data saved to %s", epoch, save_path)


# ======================================================================
# IRRA Integration Example
# ======================================================================

IRRA_INTEGRATION_EXAMPLE = """
# ==================================================================
# How to integrate with IRRA (zero-difference from original code)
# ==================================================================

from modality_grad_modulator import ModalityGradModulator, ModulationConfig

# 1. Config  (use same hyperparameters as original args)
config = ModulationConfig(
    tau=int(args.tau),          # default 1
    gammb=args.gammb,           # default 1.65
    gaeta=args.gaeta,           # default 1.0
    alpha=args.alpha,           # default 0.9
    beta=args.beta,             # default 0.999
    enabled=args.modulation,    # default True
    output_dir=args.output_dir,
)

# 2. Module filters  (match original IRRA filtering logic)
def shared_filter(name, mod):
    \"\"\"Shared cross-modal modules: NOT base_model, NOT classifier, NOT ln_pre_\"\"\"
    return (not name.startswith('base_model')
            and not name.endswith('classifier')
            and not name.startswith('ln_pre_'))

def img_enc_filter(name, mod):
    return name.startswith(('base_model.visual', 'ln_pre_i'))

def txt_enc_filter(name, mod):
    return name.startswith(('base_model.transformer', 'ln_pre_t'))

# 3. Attach
modulator = ModalityGradModulator(config)
modulator.attach(model, shared_filter, img_enc_filter, txt_enc_filter)

# 4. Training loop  (minimal changes to original do_train)
for epoch in range(start_epoch, num_epoch + 1):
    modulator.on_epoch_start(model)
    model.train()

    for n_iter, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        # ---- Three forward passes ----
        modulator.pre_forward(model)

        ret = model(batch)
        modulator.capture('normal')

        with torch.no_grad():
            e_txt_ret = model(batch.copy(), erase_modality="e_txt")
            modulator.capture('e_txt')

            e_img_ret = model(batch.copy(), erase_modality="e_img")
            modulator.capture('e_img')

        # ---- Backward ----
        total_loss = sum([v for k, v in ret.items() if "loss" in k])
        optimizer.zero_grad()
        total_loss.backward()

        # ---- Modulate gradients ----
        modulator.post_backward(
            model, step=n_iter, data_len=len(train_loader),
            batch_size=batch['images'].shape[0],
            all_loss=ret['mlm_loss'],
            e_txt_loss=e_txt_ret['mlm_loss'],
            e_img_loss=e_img_ret['mlm_loss'],
        )

        del e_txt_ret, e_img_ret
        torch.cuda.empty_cache()

        optimizer.step()
        synchronize()

    stats = modulator.on_epoch_end(model, epoch)
    scheduler.step()

# ==================================================================
# How to use with OTHER multimodal models  (e.g. CLIP, BLIP, etc.)
# ==================================================================

# Step 1: Identify the three module categories in YOUR model:
#   - shared_filter:  Cross-modal interaction layers (hooks + neuron-level pen)
#   - img_enc_filter: Image encoder leaves (encoder-level suppression)
#   - txt_enc_filter: Text encoder leaves (encoder-level suppression)
#
# Step 2: Add an ``erase_modality`` mechanism to your model's forward():
#   - 'e_txt': mix text features with 50% Gaussian noise
#   - 'e_img': mix image features with 50% Gaussian noise
#   (The noise ratio 0.5 is hardcoded in the original IRRA. If you want to
#   tune it, add it as a parameter.)
#
# Step 3: Choose which loss to use for modulation.
#   The original IRRA uses ``mlm_loss`` (the cross-modal MLM loss), NOT the
#   total loss.  Pick a loss that depends on BOTH modalities for best results.
#
# Step 4: Follow the lifecycle shown in the IRRA example above.

# Example filter for a CLIP-like model:
# def shared_filter(name, mod):
#     return 'cross_attn' in name or 'fusion' in name
# def img_enc_filter(name, mod):
#     return name.startswith('visual.')
# def txt_enc_filter(name, mod):
#     return name.startswith('text.')
"""
