# train_medsam_idrid.py  (tqdm + CSV logging + progress plot)
import os, json, random, argparse, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import distance_transform_edt
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry

# -------------------- Dataset --------------------
class IDRiDNPY(Dataset):
    def __init__(self, root, meta_json=None, split="train", val_frac=0.1, seed=0):
        root = Path(root)
        if meta_json is None:
            meta_json = root / "meta.json"
        with open(meta_json) as f:
            meta = json.load(f)

        keys = sorted([m["img"] for m in meta])
        n_val = max(1, int(len(keys) * val_frac))
        val_keys = set(keys[-n_val:])  # last chunk as val

        if split == "train":
            self.meta = [m for m in meta if m["img"] not in val_keys]
        else:
            self.meta = [m for m in meta if m["img"] in val_keys]
        self.root = root

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        m = self.meta[i]
        img = np.load(m["img"]).astype(np.float32)      # (1024,1024,3) in [0,1]
        gt  = np.load(m["gt"]).astype(np.uint8)         # (1024,1024) {0,1}
        img_t = torch.from_numpy(img).permute(2, 0, 1)  # (3,1024,1024)
        gt_t  = torch.from_numpy(gt)[None, ...].float() # (1,1024,1024)
        return img_t, gt_t, m

# -------------------- Prompt helpers --------------------
def to_2d_mask(mask: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask > thresh
    m = mask.squeeze()
    while m.ndim > 2:
        m = m.any(dim=0)
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze/merge, got {tuple(m.shape)}")
    return m

def bbox_from_mask(mask: torch.Tensor, thresh: float = 0.5):
    m = to_2d_mask(mask, thresh=thresh)
    ys, xs = torch.nonzero(m, as_tuple=True)
    if ys.numel() == 0:
        return None
    x0 = int(xs.min().item()); y0 = int(ys.min().item())
    x1 = int(xs.max().item()); y1 = int(ys.max().item())
    return [x0, y0, x1, y1]

def sample_pos(mask: torch.Tensor, k: int = 1, thresh: float = 0.5):
    m = to_2d_mask(mask, thresh=thresh)
    ys, xs = torch.nonzero(m, as_tuple=True)
    n = xs.numel()
    if n == 0:
        return []
    k = min(k, n)
    idx = torch.randperm(n)[:k]
    return [[int(xs[j]), int(ys[j])] for j in idx]

def sample_neg(mask: torch.Tensor, k: int = 4, thresh: float = 0.5):
    m = to_2d_mask(mask, thresh=thresh)
    bg = (~m).cpu().numpy().astype(np.uint8)
    total_bg = int(bg.sum())
    if total_bg == 0 or k <= 0:
        return []
    dist = distance_transform_edt(bg)
    flat = dist.ravel()
    k = min(k, total_bg)
    idxs = np.argpartition(flat, -k)[-k:]
    ys, xs = np.unravel_index(idxs, dist.shape)
    order = np.argsort(dist[ys, xs])[::-1]
    ys, xs = ys[order], xs[order]
    return [[int(x), int(y)] for y, x in zip(ys, xs)]

def make_prompt(mask: torch.Tensor):
    strat = random.choice(["box", "points3", "points1neg4", "box+point1"])
    box = None; pts = None

    if strat in ["box", "box+point1"]:
        b = bbox_from_mask(mask)
        if b is not None:
            box = np.array([b], dtype=np.float32)  # (1,4)

    if strat in ["points3", "points1neg4", "box+point1"]:
        pos = sample_pos(mask, k=3 if strat == "points3" else 1)
        neg = sample_neg(mask, k=4 if strat == "points1neg4" else 0)
        all_pts, labs = [], []
        if len(pos) > 0:
            all_pts.extend(pos); labs.extend([1] * len(pos))
        if len(neg) > 0:
            all_pts.extend(neg); labs.extend([0] * len(neg))
        if len(all_pts) > 0:
            pts = (
                np.array(all_pts, dtype=np.float32)[None, ...],  # (1,N,2)
                np.array(labs, dtype=np.int64)[None, ...],       # (1,N)
            )
    return strat, box, pts

# -------------------- Loss --------------------
class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, eps=1e-7):
        super().__init__()
        self.a, self.b, self.eps = alpha, beta, eps

    def forward(self, pred, target):
        p = pred.view(pred.size(0), -1)
        t = target.view(target.size(0), -1)
        tp = (p * t).sum(dim=1)
        fp = (p * (1 - t)).sum(dim=1)
        fn = ((1 - p) * t).sum(dim=1)
        tversky = (tp + self.eps) / (tp + self.a * fp + self.b * fn + self.eps)
        return 1.0 - tversky.mean()

# -------------------- Plot helper --------------------
def save_progress_plot(history, out_path_png, out_path_svg=None):
    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_dice = history["val_dice"]

    plt.figure(figsize=(8, 5))
    ax1 = plt.gca()
    line1, = ax1.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    line2, = ax2.plot(epochs, val_dice, label="Val Dice", linewidth=2)
    ax2.set_ylabel("Val Dice")

    # shared legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc="lower right")
    plt.title("MedSAM IDRiD Finetuning Progress")

    plt.tight_layout()
    plt.savefig(out_path_png, dpi=150)
    if out_path_svg:
        plt.savefig(out_path_svg)
    plt.close()

# -------------------- Train --------------------
def train(args):
    device = torch.device(args.device)
    ds_train = IDRiDNPY(args.data_root, split="train", val_frac=args.val_frac, seed=0)
    ds_val   = IDRiDNPY(args.data_root, split="val",   val_frac=args.val_frac, seed=0)

    train_loader = DataLoader(ds_train, batch_size=1, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(ds_val,   batch_size=1, shuffle=False, num_workers=2)
    print(f"Train={len(ds_train)}  Val={len(ds_val)}")

    sam = sam_model_registry["vit_b"](checkpoint=args.checkpoint).to(device)
    sam.train()
    if args.freeze_image_encoder:
        for p in sam.image_encoder.parameters():
            p.requires_grad = False

    params = [p for p in sam.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    loss_fn = TverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV logging setup
    csv_path = out_dir / "progress_lrNew.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_dice", "lr"])

    history = {"epoch": [], "train_loss": [], "val_dice": []}
    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        # -------------------- Training --------------------
        sam.train()
        tot_loss = 0.0
        n_batches = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for img_t, gt_t, _ in train_bar:
            img_t = img_t.to(device)  # (1,3,1024,1024)
            gt_t  = gt_t.to(device)   # (1,1,1024,1024)

            with torch.no_grad():
                img_embed = sam.image_encoder(img_t)  # (1,256,64,64)

            strat, box, pts = make_prompt(gt_t)  # prompts from GT

            box_t = None; points_t = None
            if box is not None:
                box_t = torch.as_tensor(box, dtype=torch.float32, device=device)[:, None, :]  # (1,1,4)
            if pts is not None:
                coords = torch.as_tensor(pts[0], dtype=torch.float32, device=device)  # (1,N,2)
                labels = torch.as_tensor(pts[1], dtype=torch.int64,   device=device)  # (1,N)
                points_t = (coords, labels)

            sparse, dense = sam.prompt_encoder(points=points_t, boxes=box_t, masks=None)
            logits_low, _ = sam.mask_decoder(
                image_embeddings=img_embed,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            prob = torch.sigmoid(logits_low)  # (1,1,256,256)
            prob_up = F.interpolate(
                prob, size=(gt_t.shape[-2], gt_t.shape[-1]),
                mode="bilinear", align_corners=False
            )

            loss = loss_fn(prob_up, gt_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

            n_batches += 1
            tot_loss += float(loss.item())
            avg_loss = tot_loss / n_batches
            train_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}",
                                  lr=opt.param_groups[0]["lr"])

        mean_train_loss = tot_loss / max(1, n_batches)

        # -------------------- Validation --------------------
        sam.eval()
        dices = []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]  ", leave=False)
        with torch.no_grad():
            for img_t, gt_t, _ in val_bar:
                img_t = img_t.to(device); gt_t = gt_t.to(device)
                embed = sam.image_encoder(img_t)

                b = bbox_from_mask(gt_t)
                if b is None:
                    dices.append(0.0)
                    val_bar.set_postfix(mean_dice=f"{np.mean(dices):.4f}")
                    continue

                box = torch.as_tensor(np.array([b], dtype=np.float32), device=device)[:, None, :]
                sparse, dense = sam.prompt_encoder(points=None, boxes=box, masks=None)
                logit, _ = sam.mask_decoder(
                    image_embeddings=embed,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                )
                p = torch.sigmoid(logit)
                p = F.interpolate(p, size=(gt_t.shape[-2], gt_t.shape[-1]),
                                  mode="bilinear", align_corners=False)
                pred = (p > 0.5).float()

                inter = (pred * gt_t).sum()
                denom = pred.sum() + gt_t.sum()
                dice = float((2 * inter + 1e-7) / (denom + 1e-7))
                dices.append(dice)
                val_bar.set_postfix(mean_dice=f"{np.mean(dices):.4f}")

        mean_dice = float(np.mean(dices)) if len(dices) > 0 else 0.0
        tqdm.write(f"Epoch {epoch:03d} | train_loss={mean_train_loss:.4f} | val_dice={mean_dice:.4f}")

        # ---- Log to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{mean_train_loss:.6f}", f"{mean_dice:.6f}", f"{opt.param_groups[0]['lr']:.6g}"])

        # ---- Update in-memory history and save plot
        history["epoch"].append(epoch)
        history["train_loss"].append(mean_train_loss)
        history["val_dice"].append(mean_dice)
        save_progress_plot(history, out_dir / "progress_lrNew.png", out_dir / "progress_lrNew.svg")

        # ---- Save best
        if mean_dice > best_dice:
            best_dice = mean_dice
            ckpt_path = out_dir / "medsam_idrid_best_lrNew.pth"
            torch.save(sam.state_dict(), ckpt_path)
            tqdm.write(f"[OK] Saved best to {ckpt_path} (dice {best_dice:.4f})")

    # final save
    final_path = out_dir / "medsam_idrid_last_lrNew.pth"
    torch.save(sam.state_dict(), final_path)
    tqdm.write(f"[OK] Saved last to {final_path}")
    tqdm.write(f"[OK] CSV logged at: {csv_path}")
    tqdm.write(f"[OK] Plots saved: {out_dir/'progress.png'} and {out_dir/'progress.svg'}")

# -------------------- CLI --------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/npy/IDRiD_Seg")
    ap.add_argument("--checkpoint", default="work_dir/MedSAM/medsam_vit_b.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--freeze_image_encoder", action="store_true", default=True)
    ap.add_argument("--tversky_alpha", type=float, default=0.7)
    ap.add_argument("--tversky_beta",  type=float, default=0.3)
    ap.add_argument("--out_dir", default="runs/medsam_idrid_ft")
    args = ap.parse_args()
    train(args)
