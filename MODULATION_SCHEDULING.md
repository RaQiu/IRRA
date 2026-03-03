# Cross-Modal Adaptive Gradient Modulation - Scheduling Document

## 1. Overview

The modulation plugin (`modality_grad_modulator.py`) performs adaptive gradient modification
based on the modal specificity of neurons and the relative learning progress of each modality.
It operates within the training loop at precisely defined timing points.

---

## 2. Lifecycle API

```
modulator = ModalityGradModulator(config)
modulator.attach(model, shared_filter, img_enc_filter, txt_enc_filter)

for epoch:
    modulator.on_epoch_start(model)
    for step, batch in loader:
        modulator.pre_forward(model)           # [T1] Register hooks
        ret = model(batch)                     #      Normal forward
        modulator.capture('normal')            # [T2] Snapshot activations
        e_txt_ret = model(batch, 'e_txt')      #      Text-erased forward
        modulator.capture('e_txt')             # [T3] Snapshot activations
        e_img_ret = model(batch, 'e_img')      #      Image-erased forward
        modulator.capture('e_img')             # [T4] Snapshot activations
        loss.backward()                        #      Compute gradients
        modulator.post_backward(model, ...)    # [T5] Modify gradients
        optimizer.step()                       #      Update parameters
    stats = modulator.on_epoch_end(model, epoch) # [T6] Save data & reset
```

---

## 3. Timing Points Detail

### T1: `pre_forward(model)` -- Before the first forward pass

**What happens:**
- Registers `forward_hook` on all shared (cross-modal) modules
- Hooks capture `input[0].detach()` from each module's forward call

**When to call:** Once per step, before the normal forward pass.
Hooks remain active through all three forward passes (normal, e_txt, e_img).

### T2/T3/T4: `capture(key)` -- After each forward pass

**What happens:**
- Copies the activation dict `{id(module): tensor}` under the given key
- Clears the activation buffer for the next forward pass

**Keys:** `'normal'`, `'e_txt'`, `'e_img'` (order matters)

### T5: `post_backward(model, step, data_len, batch_size, all_loss, e_txt_loss, e_img_loss)`

This is the core scheduling point. It contains multiple sub-phases:

```
post_backward()
  |
  +-- [Phase A] Fig1(c) noise experiment (first step of epoch only)
  |     Runs REGARDLESS of config.enabled
  |
  +-- [Phase B] If not config.enabled: cleanup & return
  |
  +-- [Phase C] _modulate() -- Core gradient modification
        |
        +-- [C1] model.eval()
        +-- [C2] Loss delta accumulation (every step)
        +-- [C3] Modal specificity check (every tau steps)
        +-- [C4] Gradient modification (every data_len//2 steps)
        |     |
        |     +-- [C4a] DDP sync (if multi-GPU)
        |     +-- [C4b] Compute iscore_txt, iscore_img from loss deltas
        |     +-- [C4c] Per-module pen vector computation
        |     +-- [C4d] Sigmoid stability modulation (Eqn.9)
        |     +-- [C4e] Apply pen to shared module gradients
        |     +-- [C4f] Encoder-level suppression (img & txt encoders)
        |     +-- [C4g] Reset loss accumulators
        |
        +-- [C5] Remove hooks
        +-- [C6] model.train()
```

### T6: `on_epoch_end(model, epoch)` -- After the last step of the epoch

**What happens:**
- Saves Fig1(c) gradient data to pickle file (rank 0 only)
- Logs per-module and global modal neuron counts
- Logs gradient ratio statistics
- Resets: modal counts, grad ratio stats, grad cache

---

## 4. Frequency Table

| Operation                         | Frequency                   | Controlled By        |
|-----------------------------------|-----------------------------|----------------------|
| Hook registration & removal       | Every step                  | -                    |
| Activation capture (3x)           | Every step                  | -                    |
| Loss delta accumulation           | Every step                  | -                    |
| Modal specificity check           | Every `tau` steps           | `config.tau`         |
| Fig1(c) noise experiment          | First step per epoch        | `config.fig1c_enabled` |
| **Gradient modification**         | **Every `data_len//2` steps** | Half-epoch boundary |
| DDP synchronization               | Every `data_len//2` steps   | Auto (if DDP)        |
| Sigmoid stability modulation      | When `times >= gaeta`       | `config.gaeta`       |
| Epoch-end stats & reset           | Every epoch                 | -                    |

---

## 5. Module Categories

The plugin operates on three categories of modules, defined by user-provided filter functions:

### Shared Modules (cross-modal layers)
- **Hooks:** YES (activation capture)
- **Neuron-level pen:** YES (gradient scaling per neuron based on modal specificity)
- **Example (IRRA):** `cross_attn`, `cross_modal_transformer.*`, `mlm_head.*`
- **Filter:** `not base_model, not classifier, not ln_pre_`

### Image Encoder Modules
- **Hooks:** NO
- **Encoder-level suppression:** YES (`param.grad *= (1 - iscore_img)`)
- **Fig1(c) gradient collection:** YES
- **Example (IRRA):** `base_model.visual.*`, `ln_pre_i`

### Text Encoder Modules
- **Hooks:** NO
- **Encoder-level suppression:** YES (`param.grad *= (1 - iscore_txt)`)
- **Fig1(c) gradient collection:** YES
- **Example (IRRA):** `base_model.transformer.*`, `ln_pre_t`

---

## 6. Data Flow

### 6.1 Modal Specificity Detection (C3)

```
                   Normal activation
                         |
             +-----------+-----------+
             |                       |
     |normal - e_img|         |normal - e_txt|
     = delta_img               = delta_txt
             |                       |
         mean over                mean over
        batch & seq              batch & seq
             |                       |
             v                       v
     delta_img (D,)           delta_txt (D,)
             |                       |
             +------> compare <------+
                         |
              indicate vector (D,)
              -1=init, 0=txt, 1=img
                         |
         +--- accumulate into ---+
         |                       |
  modal_img_counts[name]  modal_txt_counts[name]
```

### 6.2 Loss-Based Scoring (C4b)

```
  e_img_loss - all_loss  --->  delta_img_loss (accumulated)
  e_txt_loss - all_loss  --->  delta_txt_loss (accumulated)
                  |
         divide by tot
                  |
  score = [delta_txt/tot, delta_img/tot]
                  |
           softmax(score)
                  |
              ratio (2,)
                  |
      (ratio - r_min)^gammb
                  |
           iscore (2,)
                  |
    iscore_txt       iscore_img
```

### 6.3 Pen Vector Construction (C4c + C4d)

```
  modal_img_counts > modal_txt_counts  --> pen += iscore_img
  modal_img_counts < modal_txt_counts  --> pen += iscore_txt
                  |
         [Sigmoid Stability, if times >= gaeta]
                  |
          R = gaeta * DIFF / times
          f(R) = 2/(1+exp(-alpha*(R-1))) - 1
          g(n) = 1/(1+beta*n)
          delta_z = k * f(R) * g(n)
          pen = where(pen!=0, pen+delta_z, pen)
                  |
          pen = 1 - pen
          pen = clamp(pen, 0, 1)
                  |
          Apply to gradients:
          - 1D params: grad *= pen
          - >1D params: (quirk) size check always False, pen not applied
```

### 6.4 Encoder-Level Suppression (C4f)

```
  For each image encoder leaf module:
      param.grad *= (1 - iscore_img)

  For each text encoder leaf module:
      param.grad *= (1 - iscore_txt)
```

This is a coarser, whole-encoder scaling complementing the neuron-level pen.

---

## 7. Hyperparameters

| Parameter  | Default | Description                                              |
|------------|---------|----------------------------------------------------------|
| `tau`      | 1       | Modal specificity check interval (steps)                 |
| `gammb`    | 1.65    | Loss-difference exponent for score amplification         |
| `gaeta`    | 1.0     | Sigmoid activation threshold AND oscillation weight eta  |
| `alpha`    | 0.9     | Sigmoid steepness in f(R)                                |
| `beta`     | 0.999   | Training progress decay rate in g(n)                     |
| `enabled`  | True    | Master switch for gradient modulation                    |
| `fig1c_enabled` | True | Fig1(c) gradient noise experiment                   |
| `fig1c_noise_std` | 0.1 | Gaussian noise std for Fig1(c)                      |

**Note:** In IRRA, `alpha` and `beta` are shared with the Adam optimizer arguments.
This is by design in the original code.

---

## 8. DDP (Multi-GPU) Behavior

At each half-epoch modulation point, `_sync_distributed()` performs:

1. **All-reduce modal counts** (`MODAL_IMG_COUNTS`, `MODAL_TXT_COUNTS`) via `SUM`
2. **All-reduce DIFF** (oscillation counters) via `SUM`
3. **All-reduce loss deltas** (`delta_img_loss`, `delta_txt_loss`, `tot`) via `SUM`

This ensures every GPU computes an identical `pen` vector and applies identical gradient
modifications, keeping DDP parameters in sync. Without this, pen vectors would diverge
across GPUs, causing DDP parameter mismatch.

On single-GPU, `_sync_distributed()` is a no-op.

---

## 9. Integration Checklist for New Models

1. **Define three module filters** (shared, img_enc, txt_enc)
2. **Add `erase_modality` to model's forward()** - inject 50% Gaussian noise into one modality's features
3. **Choose the cross-modal loss** for modulation (NOT total loss; pick one that depends on both modalities)
4. **Follow the lifecycle** (attach -> per-epoch -> per-step -> epoch-end)
5. **Keep rho_t separate** if you want Q/K gradient ratio monitoring (model-specific)

---

## 10. Known Quirks (Preserved for Zero-Difference)

1. **pen.size() comparison**: `param.grad.size()[1] == pen.size()` compares `int` to `torch.Size`,
   which is always `False`. So pen only applies to 1D parameters in practice.
2. **LAST overwrite**: `self._last[name] = now` is inside the `for param` loop, so it gets
   overwritten per parameter. The final value is always the same `now` vector.
3. **Gradient ratio tracking**: `GRAD_RATIO` accumulates but is never used for modulation;
   it's purely diagnostic logging.
