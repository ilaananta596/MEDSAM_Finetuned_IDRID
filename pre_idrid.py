# pre_idrid.py
import os, json, argparse
from pathlib import Path
import numpy as np
from PIL import Image
from skimage import transform, measure, morphology

CLASSES = ["MA","HE","EX","SE","OD"]
CLASS_TO_SUBDIR = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
    "OD": "5. Optic Disc",
}

def load_mask_safe(path):
    with Image.open(path) as im:
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        im = im.convert("L")
        m = np.array(im)
    return (m > 0).astype(np.uint8)

def resize_img_1024(img_np):
    # img_np: HxWx{1,3}
    if img_np.ndim == 2:
        img_np = np.repeat(img_np[:, :, None], 3, axis=-1)
    H, W, _ = img_np.shape
    img_1024 = transform.resize(
        img_np, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.float32)
    # per-image min-max → [0,1]
    img_1024 = (img_1024 - img_1024.min()) / max(1e-8, img_1024.max() - img_1024.min())
    return img_1024

def resize_mask_1024(mask_np):
    m = transform.resize(
        mask_np.astype(np.uint8),
        (1024, 1024),
        order=0, preserve_range=True, anti_aliasing=False
    ).astype(np.uint8)
    return (m > 0).astype(np.uint8)

def pre_idrid(seg_root, split, out_root, min_area=50):
    seg_root = Path(seg_root)
    imgs_dir = seg_root/"1. Original Images"/("a. Training Set" if split=="train" else "b. Testing Set")
    gts_base = seg_root/"2. All Segmentation Groundtruths"

    out_imgs = Path(out_root)/"imgs_test"; out_imgs.mkdir(parents=True, exist_ok=True)
    out_gts  = Path(out_root)/"gts_test" ; out_gts.mkdir(parents=True, exist_ok=True)
    meta = []

    pids = sorted([p.stem for p in imgs_dir.glob("*.jpg")])
    for pid in pids:
        img_path = imgs_dir/f"{pid}.jpg"
        img_np = np.array(Image.open(img_path).convert("RGB"))
        img_1024 = resize_img_1024(img_np)

        for c in CLASSES:
            mpath = gts_base/("a. Training Set" if split=="train" else "b. Testing Set")/CLASS_TO_SUBDIR[c]/f"{pid}_{c}.tif"
            if not mpath.exists(): 
                continue
            mask = load_mask_safe(mpath)
            mask_1024 = resize_mask_1024(mask)
            if mask_1024.sum() == 0:
                continue

            # connected components on resized mask
            lab = measure.label(mask_1024, connectivity=1)
            props = measure.regionprops(lab)
            inst_id = 0
            for pr in props:
                if pr.area < min_area:
                    continue
                inst_mask = (lab == pr.label).astype(np.uint8)

                img_name = f"{pid}-{c}-{str(inst_id).zfill(3)}.npy"
                np.save(out_imgs/img_name, img_1024)
                np.save(out_gts/img_name, inst_mask)
                meta.append({
                    "pid": pid, "class": c, "instance": inst_id,
                    "img": str(out_imgs/img_name), "gt": str(out_gts/img_name),
                    "area": int(pr.area)
                })
                inst_id += 1

    with open(Path(out_root,"meta_test.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {len(meta)} samples to {out_root}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_root", required=True,
        help='.../A. Segmentation')
    ap.add_argument("--split", choices=["train","test"], default="est")
    ap.add_argument("--out_root", default="data/npy/IDRiD_Seg")
    ap.add_argument("--min_area", type=int, default=50)
    args = ap.parse_args()
    pre_idrid(args.seg_root, args.split, args.out_root, args.min_area)
