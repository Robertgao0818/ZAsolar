"""Boundary-aware Mask R-CNN loss patch.

Replaces ``torchvision.models.detection.roi_heads.maskrcnn_loss`` with a
variant that supports:

  1. Per-pixel ignore band: skips BCE on a band straddling the polygon edge.
  2. Per-instance mask weight: scales (or zeros out) the BCE for entire
     instances. ``mask_weight=0`` means an instance is dropped from mask
     supervision entirely (still contributes to box + cls).
  3. Per-pixel soft mask weight: float multiplier (0..1) applied to BCE
     pixel-wise. Used for boundary-band per-source weighting (e.g.,
     reviewed-prediction GT gets weight=0 on the boundary band, weight=1
     on the foreground/background interior, so its halo bias does not
     supervise mask edges).

Mechanism: torchvision's RoIHeads.forward looks up the loss function via
``roi_heads.maskrcnn_loss`` at call time, so replacing the module-level
attribute is enough — no subclass / forward override needed.

The patch reads the per-image ignore_masks and mask_weights from a
module-level batch state because torchvision's RoIHeads.forward only forwards
``t["masks"]`` and ``t["labels"]`` to the loss fn — the custom supervision
fields have to be smuggled in out-of-band.

How the batch state is populated (production contract — NOT a manual
``stash_batch_supervision`` call in the training loop): stashing happens
automatically inside the ``model.transform.forward`` wrapper installed by
``install_transform_aux_resize``. That wrapper is the only place that sees the
*post-transform* targets (resized to torchvision's internal spatial range),
which is exactly the shape the patched mask loss needs. The training loop just
runs ``model(images, targets)``.

Lifecycle seam — use ``MaskSupervisionPatch`` rather than wiring the three
patches + the per-batch boundary by hand (install order matters, and the
stale-state hazard below is invisible in a scattered set of calls):

    patch = MaskSupervisionPatch(
        model, enable_aux_resize=True, enable_box_loss_telemetry=False,
    )
    patch.install()                 # mask-loss patch + transform wrapper (+ box telemetry)
    for images, targets in loader:
        with patch.batch():         # asserts a fresh stash happened this forward
            loss_dict = model(images, targets)
    patch.teardown()                # restore torchvision + unwrap the model

Stale-state hazard the seam guards: the auto-stash overwrites *all* batch
state on every training forward, so the loss is correct as long as the stash
runs. The state is never cleared between batches. If a training forward ever
computed the mask loss without a fresh stash (transform not in ``.training``,
or a target without ``"masks"``), the previous batch's supervision would
silently feed the loss. That path is unreachable in today's loop;
``patch.batch()`` makes it *structurally* unreachable by raising
``StaleSupervisionError`` instead of leaking.

The low-level ``install_patch`` / ``stash_batch_supervision`` /
``clear_batch_supervision`` functions remain the primitives the class delegates
to (and the existing smoke tests in ``scripts/training/jhb_phaseA/`` call them
directly); ``MaskSupervisionPatch`` is the coordinating seam over them.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torchvision.models.detection import roi_heads as _rh
from torchvision.ops import roi_align


class StaleSupervisionError(RuntimeError):
    """Raised when the patched mask loss would run against batch-supervision
    state that was not refreshed for the current forward — i.e. the per-batch
    auto-stash was skipped and the previous batch's state would leak. See the
    module docstring's stale-state note and ``MaskSupervisionPatch.batch``."""


_BATCH_STATE: dict = {
    "ignore_masks": None,
    "mask_weights": None,
    "mask_pixel_weights": None,
    "label_sources": None,
}

# Monotonic count of stash events. ``patched_maskrcnn_loss`` never reads it, so
# bumping it is byte-equivalent w.r.t. the loss; ``MaskSupervisionPatch.batch``
# samples it before/after a forward to prove a fresh stash happened this batch
# — the structural guard against the stale-state hazard documented above.
_STASH_COUNTER = 0


def stash_batch_supervision(targets: list[dict]) -> None:
    """Stash per-image supervision tensors for the patched mask loss.

    In production this is called automatically by the transform wrapper
    (``install_transform_aux_resize``), not from the training loop. It fully
    overwrites every key of ``_BATCH_STATE`` on each call, so stale state from a
    prior batch cannot survive a stash that actually runs (see the module
    docstring's stale-state note for the one path the seam still guards).
    """
    global _STASH_COUNTER
    _BATCH_STATE["ignore_masks"] = [t.get("ignore_masks") for t in targets]
    _BATCH_STATE["mask_weights"] = [t.get("mask_weights") for t in targets]
    _BATCH_STATE["mask_pixel_weights"] = [t.get("mask_pixel_weights") for t in targets]
    _BATCH_STATE["label_sources"] = [t.get("label_sources", []) for t in targets]
    _STASH_COUNTER += 1


def stash_counter() -> int:
    """Current value of the monotonic stash counter (see ``_STASH_COUNTER``)."""
    return _STASH_COUNTER


def clear_batch_supervision() -> None:
    _BATCH_STATE["ignore_masks"] = None
    _BATCH_STATE["mask_weights"] = None
    _BATCH_STATE["mask_pixel_weights"] = None
    _BATCH_STATE["label_sources"] = None


def _project_masks_on_boxes(
    masks: torch.Tensor,
    boxes: torch.Tensor,
    matched_idxs: torch.Tensor,
    M: int,
) -> torch.Tensor:
    """Same as torchvision's project_masks_on_boxes but reusable for ignore masks."""
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    masks = masks[:, None].to(rois)
    return roi_align(masks, rois, (M, M), 1.0)[:, 0]


def patched_maskrcnn_loss(
    mask_logits: torch.Tensor,
    proposals: list[torch.Tensor],
    gt_masks: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    mask_matched_idxs: list[torch.Tensor],
) -> torch.Tensor:
    """Boundary-aware BCE: per-pixel ignore + per-instance weight."""
    M = mask_logits.shape[-1]

    labels = [gl[idxs] for gl, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        _project_masks_on_boxes(m, p, i, M)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    weights_per_img = _BATCH_STATE.get("mask_weights")
    has_weights = (
        weights_per_img is not None
        and any(w is not None for w in weights_per_img)
    )
    if has_weights:
        per_prop_w = []
        for w_full, idxs in zip(weights_per_img, mask_matched_idxs):
            if w_full is None or len(w_full) == 0:
                per_prop_w.append(torch.ones(len(idxs), device=mask_logits.device))
            else:
                w_full_dev = w_full.to(idxs.device)
                per_prop_w.append(w_full_dev[idxs].to(mask_logits.device))
        per_prop_w = torch.cat(per_prop_w, dim=0) if per_prop_w else None
    else:
        per_prop_w = None

    ignores_per_img = _BATCH_STATE.get("ignore_masks")
    has_ignores = (
        ignores_per_img is not None
        and any(im is not None for im in ignores_per_img)
    )
    if has_ignores:
        ig_targets = []
        for im, p, i in zip(ignores_per_img, proposals, mask_matched_idxs):
            if im is None or len(im) == 0:
                ig_targets.append(torch.zeros(len(p), M, M, device=p.device))
            else:
                ig_targets.append(_project_masks_on_boxes(im, p, i, M))
        ig_targets = torch.cat(ig_targets, dim=0) if ig_targets else None
    else:
        ig_targets = None

    pix_weights_per_img = _BATCH_STATE.get("mask_pixel_weights")
    has_pix_weights = (
        pix_weights_per_img is not None
        and any(pw is not None for pw in pix_weights_per_img)
    )
    if has_pix_weights:
        pix_w_targets = []
        for pw, p, i in zip(pix_weights_per_img, proposals, mask_matched_idxs):
            if pw is None or len(pw) == 0:
                pix_w_targets.append(torch.ones(len(p), M, M, device=p.device))
            else:
                pix_w_targets.append(_project_masks_on_boxes(pw, p, i, M))
        pix_w_targets = torch.cat(pix_w_targets, dim=0) if pix_w_targets else None
    else:
        pix_w_targets = None

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    selected_logits = mask_logits[
        torch.arange(labels.shape[0], device=labels.device), labels
    ]
    bce_per_pixel = F.binary_cross_entropy_with_logits(
        selected_logits, mask_targets, reduction="none"
    )

    if ig_targets is not None:
        valid_pixel = (ig_targets <= 0.5).to(bce_per_pixel.dtype)
    else:
        valid_pixel = torch.ones_like(bce_per_pixel)

    if pix_w_targets is not None:
        soft_w = pix_w_targets.clamp(0.0, 1.0).to(bce_per_pixel.dtype)
        valid_pixel = valid_pixel * soft_w

    if per_prop_w is not None:
        w_pp = per_prop_w[:, None, None].to(bce_per_pixel.dtype)
        weighted = bce_per_pixel * valid_pixel * w_pp
        denom = (valid_pixel * w_pp).sum().clamp(min=1.0)
    else:
        weighted = bce_per_pixel * valid_pixel
        denom = valid_pixel.sum().clamp(min=1.0)

    return weighted.sum() / denom


_ORIGINAL_LOSS = _rh.maskrcnn_loss


def install_patch() -> None:
    """Replace torchvision's maskrcnn_loss. Idempotent."""
    _rh.maskrcnn_loss = patched_maskrcnn_loss


def restore_original() -> None:
    """Restore torchvision's stock maskrcnn_loss (e.g. for tests)."""
    _rh.maskrcnn_loss = _ORIGINAL_LOSS


def is_installed() -> bool:
    return _rh.maskrcnn_loss is patched_maskrcnn_loss


# ════════════════════════════════════════════════════════════════════════
# Post-transform supervision stash (with aux spatial resize)
# ════════════════════════════════════════════════════════════════════════
def _resize_aux_to_mask(aux: torch.Tensor, new_h: int, new_w: int,
                        mode: str) -> torch.Tensor:
    """Resize a per-instance spatial tensor (N, H, W) to (N, new_h, new_w).

    ``mode='bilinear'`` for soft float weights (mask_pixel_weights),
    ``mode='nearest'`` for binary masks (ignore_masks).
    """
    if aux.numel() == 0:
        return aux
    if aux.dim() == 2:
        aux = aux.unsqueeze(0)
    if aux.shape[-2] == new_h and aux.shape[-1] == new_w:
        return aux
    x = aux.float().unsqueeze(0)  # (1, N, H, W)
    kwargs: dict = {"size": (new_h, new_w), "mode": mode}
    if mode == "bilinear":
        kwargs["align_corners"] = False
    out = F.interpolate(x, **kwargs)
    return out[0].to(aux.dtype)


def install_transform_aux_resize(model) -> None:
    """Wrap ``model.transform.forward`` so that per-image auxiliary spatial
    fields (``mask_pixel_weights``, ``ignore_masks``) are resized in sync
    with ``target['masks']`` after the standard ``GeneralizedRCNNTransform``,
    then stash the supervision state for the patched mask loss.

    Why this exists: torchvision's transform resizes images and
    ``target['masks']`` from chip resolution (e.g. 400) to its internal
    spatial range (default min_size=800). Custom dict keys are passed
    through untouched. If we stashed pre-transform, the patched mask loss
    would then call ROIAlign with post-transform proposals (in 800-space)
    against pre-transform weight maps (in 400-space) → samples at 2× the
    correct coordinates → spatial mismatch. This wrapper runs after the
    standard transform, sees the post-transform mask shape, and resizes
    auxiliary weights to match before stashing.

    Idempotent.
    """
    transform = model.transform
    if getattr(transform, "_aux_resize_installed", False):
        return
    base_forward = transform.forward
    # Stash the original so teardown can restore it (clean test isolation).
    transform._aux_resize_base_forward = base_forward

    def patched_forward(images, targets=None):
        image_list, targets_out = base_forward(images, targets)
        if targets_out is not None and transform.training:
            for tgt in targets_out:
                if "masks" not in tgt:
                    continue
                new_h, new_w = tgt["masks"].shape[-2], tgt["masks"].shape[-1]
                pw = tgt.get("mask_pixel_weights")
                if pw is not None:
                    tgt["mask_pixel_weights"] = _resize_aux_to_mask(
                        pw, new_h, new_w, "bilinear",
                    )
                ig = tgt.get("ignore_masks")
                if ig is not None:
                    tgt["ignore_masks"] = _resize_aux_to_mask(
                        ig, new_h, new_w, "nearest",
                    )
            stash_batch_supervision(targets_out)
        return image_list, targets_out

    transform.forward = patched_forward
    transform._aux_resize_installed = True


def uninstall_transform_aux_resize(model) -> None:
    """Restore the original ``model.transform.forward``. Idempotent."""
    transform = model.transform
    if not getattr(transform, "_aux_resize_installed", False):
        return
    base = getattr(transform, "_aux_resize_base_forward", None)
    if base is not None:
        transform.forward = base
    transform._aux_resize_installed = False
    transform._aux_resize_base_forward = None


def is_transform_aux_resize_installed(model) -> bool:
    return getattr(model.transform, "_aux_resize_installed", False)


# ════════════════════════════════════════════════════════════════════════
# Per-source box regression loss tracking
# (Signal collection for next-round box_trusted decision — does NOT alter
#  gradients. The patched fastrcnn_loss returns the same scalar; it only
#  records per-source averages into a module-level dict per call.)
# ════════════════════════════════════════════════════════════════════════
_BOX_LOSS_BUCKETS: dict[str, list[float]] = {}
_LAST_MATCHED_IDXS: list[torch.Tensor] | None = None
_LAST_LABEL_SOURCES: list[list[str]] | None = None


def reset_box_loss_buckets() -> None:
    """Call at start of each epoch."""
    _BOX_LOSS_BUCKETS.clear()


def box_loss_bucket_means() -> dict[str, float]:
    """Return mean box reg loss per source bucket (or empty if no data)."""
    out = {}
    for src, vals in _BOX_LOSS_BUCKETS.items():
        if vals:
            out[src] = float(sum(vals) / len(vals))
    return out


def stash_matched_info(matched_idxs: list[torch.Tensor], label_sources: list[list[str]]) -> None:
    """Stash per-image matched_idxs + per-image GT label_sources just before
    fastrcnn_loss runs.  matched_idxs[i] is the per-proposal → GT-index tensor
    for image i (after select_training_samples sampling)."""
    global _LAST_MATCHED_IDXS, _LAST_LABEL_SOURCES
    _LAST_MATCHED_IDXS = matched_idxs
    _LAST_LABEL_SOURCES = label_sources


def _clear_matched_info() -> None:
    global _LAST_MATCHED_IDXS, _LAST_LABEL_SOURCES
    _LAST_MATCHED_IDXS = None
    _LAST_LABEL_SOURCES = None


def patched_fastrcnn_loss(
    class_logits: torch.Tensor,
    box_regression: torch.Tensor,
    labels: list[torch.Tensor],
    regression_targets: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """torchvision's fastrcnn_loss + per-source box reg loss bucketing.

    Returns identical scalars to torchvision (no gradient impact).  Side
    effect: if matched info has been stashed, per-source means accumulate
    into _BOX_LOSS_BUCKETS for end-of-epoch readout.
    """
    labels_cat = torch.cat(labels, dim=0)
    regression_targets_cat = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels_cat)

    sampled_pos_inds_subset = torch.where(labels_cat > 0)[0]
    labels_pos = labels_cat[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression_r = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    pred_pos = box_regression_r[sampled_pos_inds_subset, labels_pos]
    tgt_pos = regression_targets_cat[sampled_pos_inds_subset]

    # per-element loss for bucketing; sum over coords per proposal
    bx_per_elem = F.smooth_l1_loss(pred_pos, tgt_pos, beta=1 / 9, reduction="none")
    bx_per_prop = bx_per_elem.sum(dim=1)  # (n_pos,)

    # canonical aggregate (matches torchvision behavior exactly)
    box_loss = bx_per_prop.sum() / labels_cat.numel()

    # ── side-effect: bucket per-source if matched info available ────────
    if _LAST_MATCHED_IDXS is not None and _LAST_LABEL_SOURCES is not None:
        try:
            # Flatten matched_idxs across images, parallel to labels_cat order.
            # labels comes per-image in the same order; same for matched_idxs.
            flat_matched = torch.cat(_LAST_MATCHED_IDXS, dim=0)
            # Build per-proposal source label aligned with labels_cat
            sources_flat: list[str] = []
            img_lens = [m.shape[0] for m in _LAST_MATCHED_IDXS]
            cursor = 0
            for img_i, n_prop in enumerate(img_lens):
                ls_img = _LAST_LABEL_SOURCES[img_i] if img_i < len(_LAST_LABEL_SOURCES) else []
                mi = _LAST_MATCHED_IDXS[img_i].tolist()
                for prop_i in range(n_prop):
                    gt_i = mi[prop_i]
                    if 0 <= gt_i < len(ls_img):
                        sources_flat.append(ls_img[gt_i])
                    else:
                        sources_flat.append("")
                cursor += n_prop

            if len(sources_flat) == labels_cat.shape[0]:
                # restrict to positive proposals only (same as box reg)
                pos_idx_list = sampled_pos_inds_subset.tolist()
                pos_losses = bx_per_prop.detach().cpu().tolist()
                for k, pi in enumerate(pos_idx_list):
                    src = sources_flat[pi] or "UNKNOWN"
                    _BOX_LOSS_BUCKETS.setdefault(src, []).append(pos_losses[k])
        except Exception as e:  # don't break training over a bucketing bug
            print(f"[WARN] per-source box loss bucketing skipped: {e}")

    return classification_loss, box_loss


_ORIGINAL_FASTRCNN_LOSS = _rh.fastrcnn_loss


def install_fastrcnn_patch() -> None:
    _rh.fastrcnn_loss = patched_fastrcnn_loss


def restore_fastrcnn() -> None:
    _rh.fastrcnn_loss = _ORIGINAL_FASTRCNN_LOSS


def is_fastrcnn_patched() -> bool:
    return _rh.fastrcnn_loss is patched_fastrcnn_loss


def wrap_select_training_samples(model) -> None:
    """Monkey-patch model.roi_heads.select_training_samples to stash
    matched_idxs (per-image, per-sampled-proposal → GT index) into module
    state for downstream per-source box-loss bucketing.

    Idempotent: re-wrapping returns without action.
    """
    rh_obj = model.roi_heads
    if getattr(rh_obj, "_select_training_samples_wrapped", False):
        return
    original = rh_obj.select_training_samples
    # Stash the original so teardown can restore it (clean test isolation).
    rh_obj._select_training_samples_original = original

    def wrapper(proposals, targets):
        result = original(proposals, targets)
        # result = (proposals, matched_idxs, labels, regression_targets)
        _, matched_idxs, _, _ = result
        label_sources = _BATCH_STATE.get("label_sources") or []
        stash_matched_info(matched_idxs, label_sources)
        return result

    rh_obj.select_training_samples = wrapper
    rh_obj._select_training_samples_wrapped = True


def unwrap_select_training_samples(model) -> None:
    """Restore the original ``roi_heads.select_training_samples``. Idempotent."""
    rh_obj = model.roi_heads
    if not getattr(rh_obj, "_select_training_samples_wrapped", False):
        return
    original = getattr(rh_obj, "_select_training_samples_original", None)
    if original is not None:
        rh_obj.select_training_samples = original
    rh_obj._select_training_samples_wrapped = False
    rh_obj._select_training_samples_original = None


# ════════════════════════════════════════════════════════════════════════
# Lifecycle seam — MaskSupervisionPatch
# ════════════════════════════════════════════════════════════════════════
class MaskSupervisionPatch:
    """Single lifecycle seam over the boundary-aware-supervision patch bundle.

    Replaces the four scattered install calls (``install_patch`` /
    ``install_transform_aux_resize`` / ``install_fastrcnn_patch`` /
    ``wrap_select_training_samples``) coordinated by implicit ordering in
    ``train.py main()`` with one object that owns the whole lifecycle::

        install()  →  for each batch:  with patch.batch(): model(...)  →  teardown()

    Two independent levers (mirroring train.py's flags exactly, so the *set* of
    installed patches is byte-equivalent to the pre-seam code):

      * ``enable_aux_resize`` — the mask-loss path: install the boundary-aware
        ``maskrcnn_loss`` patch and the transform wrapper that resizes the aux
        spatial fields and auto-stashes them. Driven by
        ``--per-source-mask-weight`` / ``--per-instance-mask-trusted`` and the
        jhb_phaseA spec.
      * ``enable_box_loss_telemetry`` — the per-source box-reg-loss diagnostic:
        *additionally* patch ``fastrcnn_loss`` and wrap
        ``select_training_samples``. Driven by ``--log-per-source-box-reg-loss``.
        It also needs the transform wrapper (to stash ``label_sources``) and
        installs the mask-loss patch, which is a no-op when no aux fields are
        present — identical to stock ``maskrcnn_loss``. This matches the
        pre-seam code, which installed both on this path too.

    Per-batch guard (the stale-state hazard — see module docstring):

      * ``assert_fresh_state`` (default ``True``) — ``batch()`` raises
        ``StaleSupervisionError`` if no fresh stash happened during the forward.
        Behaviour-preserving: on every *reachable* training path the auto-stash
        always runs, so this never fires; it only converts the latent silent
        leak into a loud error if a future change ever opens that path.
      * ``clear_after_batch`` (default ``False``) — the deliberate Step-B
        semantics change (actually clear the batch state after each forward).
        Off by default; flipping it on is a gated change to record in the
        experiment ledger, never folded silently into the behaviour-preserving
        seam.
    """

    def __init__(
        self,
        model,
        *,
        enable_aux_resize: bool,
        enable_box_loss_telemetry: bool = False,
        assert_fresh_state: bool = True,
        clear_after_batch: bool = False,
    ) -> None:
        self.model = model
        self.enable_aux_resize = enable_aux_resize
        self.enable_box_loss_telemetry = enable_box_loss_telemetry
        self.assert_fresh_state = assert_fresh_state
        self.clear_after_batch = clear_after_batch
        self._installed = False

    @property
    def _wrapper_installed(self) -> bool:
        # The transform wrapper (and thus the auto-stash) is installed whenever
        # either lever is on.
        return self.enable_aux_resize or self.enable_box_loss_telemetry

    def install(self, *, verbose: bool = True) -> "MaskSupervisionPatch":
        """Install the patches this object's levers select, in dependency order.

        Idempotent — every underlying installer guards against double-install,
        so calling ``install()`` twice (or re-installing the mask patch from a
        second lever) is a no-op.
        """
        if self._wrapper_installed:
            install_patch()
            install_transform_aux_resize(self.model)
        if self.enable_box_loss_telemetry:
            install_fastrcnn_patch()
            wrap_select_training_samples(self.model)
        self._installed = True
        if verbose:
            self._log_install()
        return self

    def _log_install(self) -> None:
        if not self._wrapper_installed:
            print("[PATCH] MaskSupervisionPatch: no supervision levers enabled (no-op)")
            return
        modes = []
        if self.enable_aux_resize:
            modes.append("aux-resize (boundary-aware mask loss)")
        if self.enable_box_loss_telemetry:
            modes.append("per-source box-reg-loss telemetry")
        print(f"[PATCH] MaskSupervisionPatch installed: {', '.join(modes)} "
              f"(assert_fresh_state={self.assert_fresh_state}, "
              f"clear_after_batch={self.clear_after_batch})")

    @contextmanager
    def batch(self):
        """Per-batch boundary. Enter before ``model(images, targets)``.

        On normal exit (Step A) assert a fresh stash ran during this forward —
        i.e. the previous batch's supervision did not leak into the loss. On
        any exit, if ``clear_after_batch`` is on (gated Step B), clear the
        batch state. A forward that raises is left un-asserted (the loss never
        ran) but still cleared.
        """
        start = _STASH_COUNTER
        try:
            yield
            if (self.assert_fresh_state and self._wrapper_installed
                    and _STASH_COUNTER == start):
                raise StaleSupervisionError(
                    "patched maskrcnn_loss ran without a fresh per-batch stash "
                    "(stash counter did not advance during the forward); the "
                    "previous batch's supervision would have leaked into the "
                    "loss. The auto-stash in the transform wrapper must run on "
                    "every training forward."
                )
        finally:
            if self.clear_after_batch:
                clear_batch_supervision()
                _clear_matched_info()

    def teardown(self) -> None:
        """Restore torchvision's stock losses and unwrap the model, leaving the
        process clean for test isolation / a second in-process training run."""
        restore_original()
        restore_fastrcnn()
        uninstall_transform_aux_resize(self.model)
        unwrap_select_training_samples(self.model)
        self._installed = False

    # ── box-loss telemetry readout (delegating convenience) ────────────────
    def reset_box_loss_buckets(self) -> None:
        reset_box_loss_buckets()

    def box_loss_bucket_means(self) -> dict[str, float]:
        return box_loss_bucket_means()
