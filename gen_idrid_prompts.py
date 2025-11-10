# gen_idrid_prompts.py
import json, os, random
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops

CLASSES = ["MA","HE","EX","SE","OD"]
CLASS_TO_SUBDIR = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
    "OD": "5. Optic Disc",
}
# replace your load_union_mask() with this
def load_union_mask(gts_base, split, pid, klass):
    from PIL import Image
    mdir = Path(gts_base)/("a. Training Set" if split=="train" else "b. Testing Set")/CLASS_TO_SUBDIR[klass]
    mpath = mdir/f"{pid}_{klass}.tif"
    if not mpath.exists():
        return None
    with Image.open(mpath) as im:
        # some IDRiD masks are RGBA/Palette/16-bit TIFF; make it single-channel
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        im = im.convert("L")   # force 1 channel
        m = np.array(im)
    return (m > 0).astype(np.uint8)


def sample_points(mask, k):
    ys, xs = np.where(mask>0)
    if len(xs)==0: return []
    idx = np.random.choice(len(xs), size=min(k,len(xs)), replace=False)
    return [[int(xs[i]), int(ys[i])] for i in idx]

def sample_negatives(bg_mask, k):
    ys, xs = np.where(bg_mask>0)
    if len(xs)==0: return []
    idx = np.random.choice(len(xs), size=min(k,len(xs)), replace=False)
    return [[int(xs[i]), int(ys[i])] for i in idx]

def mask_bbox(mask):
    ys, xs = np.where(mask>0)
    if len(xs)==0: return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def run(seg_root, split, out_dir):
    seg_root = Path(seg_root)
    imgs_dir = seg_root/"1. Original Images"/("a. Training Set" if split=="train" else "b. Testing Set")
    gts_base = seg_root/"2. All Segmentation Groundtruths"

    pids = sorted([p.stem for p in imgs_dir.glob("*.jpg")])
 
    rnd = np.random.RandomState(0)
    random.seed(0)
    results = {k:[] for k in ["points1","points3","points5","points1neg4","boxes"]}

    for pid in pids:
        img_path = str(imgs_dir/f"{pid}.jpg")
        # union of all classes to sample background
        H,W = Image.open(imgs_dir/f"{pid}.jpg").size[1], Image.open(imgs_dir/f"{pid}.jpg").size[0]
        all_mask = np.zeros((H,W), np.uint8)
        class_masks = {}
        for c in CLASSES:
            m = load_union_mask(gts_base, split, pid, c)
            if m is None: m = np.zeros((H,W), np.uint8)
            class_masks[c]=m
            all_mask |= m
        bg = (all_mask==0).astype(np.uint8)

        # build prompt entries per class (skip if class absent)
        entry_per_set = {k:{"image": img_path, "per_class":{}} for k in results.keys()}
        for c in CLASSES:
            m = class_masks[c]
            if m.sum()==0: continue
            entry_per_set["points1"]["per_class"][c] = {"type":"point","pos": sample_points(m,1)}
            entry_per_set["points3"]["per_class"][c] = {"type":"point","pos": sample_points(m,3)}
            entry_per_set["points5"]["per_class"][c] = {"type":"point","pos": sample_points(m,5)}
            entry_per_set["points1neg4"]["per_class"][c] = {
                "type":"point",
                "pos": sample_points(m,1),
                "neg": sample_negatives(bg,4)
            }
            bb = mask_bbox(m)
            if bb is not None:
                entry_per_set["boxes"]["per_class"][c] = {"type":"box","box":bb}

        for k in results:
            results[k].append(entry_per_set[k])

    os.makedirs(out_dir, exist_ok=True)
    for k, lst in results.items():
        with open(Path(out_dir,f"{k}.json"),"w") as f:
            json.dump(lst, f)
    print("Wrote:", [f"{k}.json" for k in results])

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_root", required=True)
    ap.add_argument("--split", choices=["train","test"], default="test")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    run(args.seg_root, args.split, args.out_dir)
