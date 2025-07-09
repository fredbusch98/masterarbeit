# hpo_optuna_lpips.py
import optuna, torch, random
from lpips_scoring import lpips_distance
from your_repo.mimicmotion import MimicMotionModel           # adapt path
from your_repo.dataloading   import build_dataloaders        # adapt path

def render_to_image(motion_tensor):
    """
    YOUR code here – convert a (T, J, xyz) or (T, J, rot) tensor
    into a 3-channel image (or stack of frames) in the range -1…1.
    Must return a 4-D torch tensor [B, 3, H, W] on the same device.
    """
    raise NotImplementedError

# ------------ Optuna objective ------------
def objective(trial: optuna.Trial) -> float:
    # ── 1. Sample hyper-parameters ───────────────────────────
    hparams = {
        "lr"      : trial.suggest_loguniform("lr", 1e-5, 1e-3),
        "bs"      : trial.suggest_categorical("bs", [16, 32, 64]),
        "latent"  : trial.suggest_int("latent", 64, 256, step=64),
        "freeze"  : trial.suggest_categorical("freeze_backbone", [True, False]),
        "smooth_w": trial.suggest_float("w_smooth", 0.0, 0.3),
    }

    # ── 2. Build model & data ────────────────────────────────
    model = MimicMotionModel(latent_dim=hparams["latent"],
                             smooth_loss_weight=hparams["smooth_w"])
    if hparams["freeze"]:
        model.freeze_backbone()

    model = model.cuda()          # comment out if CPU-only
    optim = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
    train_loader, val_loader = build_dataloaders(batch_size=hparams["bs"])

    # ── 3. Train for a few epochs ────────────────────────────
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            loss = model.compute_loss(batch)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Validation LPIPS every epoch
        val_lpips = validate_lpips(model, val_loader)
        trial.report(val_lpips, epoch)

        # Prune unpromising trials early
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_lpips   # <-- Optuna minimises by default
# -------------------------------------------

@torch.no_grad()
def validate_lpips(model, val_loader):
    model.eval()
    lpips_sum, n = 0.0, 0
    for batch in val_loader:
        gt_motion  = batch["gt"].cuda()
        pred_motion = model(batch["input"].cuda())

        gt_img   = render_to_image(gt_motion)
        pred_img = render_to_image(pred_motion)

        lpips_sum += lpips_distance(gt_img, pred_img) * gt_img.size(0)
        n         += gt_img.size(0)
    return lpips_sum / n

# ---------------------------------------

# lpips_scoring.py
import lpips, torch

# Pre-load the LPIPS network on first import; reuse for speed
_LPIPS = lpips.LPIPS(net='alex').eval().cuda()   # or .cpu() if no GPU

@torch.no_grad()
def lpips_distance(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    img1, img2 : 4-D tensors [B, 3, H, W] in -1…1 range.
    Returns mean LPIPS over the batch (float).
    """
    assert img1.shape == img2.shape, "Image shapes must match"
    dist = _LPIPS(img1, img2)          # shape [B,1,1,1]
    return dist.mean().item()

# -----------------------------------

if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        sampler  = optuna.samplers.TPESampler(multivariate=True),
        pruner   = optuna.pruners.MedianPruner(n_warmup_steps=3)
    )
    study.optimize(objective, n_trials=50, timeout=3*60*60)  # 50 trials or 3 h

    print("Best LPIPS:", study.best_value)
    print("Best hyper-parameters:\n", study.best_params)
    study.trials_dataframe().to_csv("optuna_results.csv", index=False)