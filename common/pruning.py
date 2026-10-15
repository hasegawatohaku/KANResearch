import math
import torch
import numpy as np

from common.layers.kanele import KANELEEdgeLayer

def kanele_layers(model):
    return [
        module
        for module in model.modules()
        if isinstance(module, KANELEEdgeLayer)
    ]

def count_kanele_connections(model):
    layers = kanele_layers(model)

    active = sum(int(layer.connection_mask.sum().item()) for layer in layers)
    total = sum(layer.connection_mask.numel() for layer in layers)

    return active, total

@torch.no_grad()
def prune_connections(layer, scores, target_sparsity):
    """
    Prune the lowest-scored active connections of one layer until
    ceil(total * (1 - target_sparsity)) connections remain active.

    As in kanele_regression.py, the last active connection of an output
    is never removed, and a pruned connection is never restored.
    """

    if target_sparsity <= 0:
        return 0

    active = layer.connection_mask.detach().cpu().numpy() > 0.5

    total = active.size
    desired_active = max(
        layer.n_out,
        int(math.ceil(total * (1.0 - target_sparsity)))
    )
    active_count = int(active.sum())

    if active_count <= desired_active:
        return 0

    # Least important connections first.
    order = np.argsort(
        scores.detach().cpu().numpy().ravel(),
        kind="stable"
    )
    row_counts = active.sum(axis=1)
    removed = 0

    for flat_index in order:
        if active_count <= desired_active:
            break

        out_index, in_index = divmod(int(flat_index), layer.n_in)

        if not active[out_index, in_index]:
            continue

        if row_counts[out_index] <= 1:
            continue

        active[out_index, in_index] = False
        row_counts[out_index] -= 1
        active_count -= 1
        removed += 1

    layer.connection_mask.copy_(
        torch.as_tensor(
            active,
            dtype=layer.connection_mask.dtype,
            device=layer.connection_mask.device
        )
    )

    return removed

@torch.no_grad()
def update_kanele_pruning(model, 
                          calibration_images, 
                          target_sparsity,
                          batch_size
    ):
    """
    Score every connection on the calibration images (inputs seen by each
    layer with the current masks) and update the masks of all layers.
    """

    layers = kanele_layers(model)
    device = next(model.parameters()).device
    was_training = model.training

    model.eval()

    for layer in layers:
        layer.begin_pruning_stats()

    try:
        for start in range(0, len(calibration_images), batch_size):
            model(calibration_images[start:start + batch_size].to(device))

        removed = 0

        for layer in layers:
            removed += prune_connections(
                layer,
                layer.connection_scores(),
                target_sparsity
            )
    finally:
        for layer in layers:
            layer.end_pruning_stats()

        model.train(was_training)

    return removed

def make_kanele_epoch_end(
    model,
    calibration,
    train_dataset,
    target_sparsity,
    pruning_start_epoch,
    epochs
):
    """
    Pruning schedule of kanele_regression.py as an epoch-end hook:
    from KANELE_PRUNING_START_EPOCH the sparsity grows linearly and
    reaches KANELE_TARGET_SPARSITY at the last epoch.
    """

    # Fixed calibration images, loaded once. A tensor is used instead of a
    # DataLoader, so that calibration does not consume the global RNG.
    count = min(calibration, len(train_dataset))
    calibration_images = torch.stack(
        [train_dataset[i][0] for i in range(count)]
    )

    def epoch_end(epoch):
        if target_sparsity <= 0 or epoch < pruning_start_epoch:
            return

        progress = (epoch - pruning_start_epoch + 1) / max(
            epochs - pruning_start_epoch + 1, 1
        )
        sparsity = target_sparsity * min(progress, 1.0)

        removed = update_kanele_pruning(model, calibration_images, sparsity)
        active, total = count_kanele_connections(model)

        if epoch == 1 or epoch % 2 == 0:
            print(f"pruning: target_sparsity={100.0 * sparsity:.1f}% "
                f"active_connections={active}/{total} "
                f"newly_pruned={removed}"
            )

    return epoch_end