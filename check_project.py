"""
NeuroSeg honest self-check.  Put this file in the PROJECT ROOT (next to src/) and run:

    python check_project.py                 # 3 held-out cases
    python check_project.py --max-cases 1   # quicker

It never prints a "ready" slogan: each check reports PASS / WARN / FAIL, and the
targets are reported separately as MET / NOT MET using real held-out scores.
"""
import os, sys, json, random, argparse, time, traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

ap = argparse.ArgumentParser()
ap.add_argument("--max-cases", type=int, default=3)
args = ap.parse_args()

results = []


def rec(status, name, detail=""):
    results.append(status)
    print(f"[{status:<4}] {name}" + (f"  -  {detail}" if detail else ""), flush=True)


DICE_T = {"WT": 0.88, "ET": 0.82}
HD95_T = 5.0
CKPT = os.path.join(ROOT, "outputs", "checkpoints", "best_segresnet_metric.pth")
HIST = os.path.join(ROOT, "outputs", "checkpoints", "training_history.json")

print("=" * 72)
print(" NEUROSEG SELF-CHECK")
print("=" * 72)

# ---------------------------------------------------------------- imports
try:
    # pyrefly: ignore[missing-import]
    import numpy as np
    # pyrefly: ignore[missing-import]
    import torch
    # pyrefly: ignore[missing-import]
    import nibabel as nib
    from src.inference import BrainTumorInferenceEngine, compute_clinical_validation_metrics
    from src.dataset_pipeline import load_brats_file_list, get_val_transforms
    from src.model import convert_labels_to_brats_classes
    from src.volumetrics import compute_subregion_volumetrics, compute_rano_criteria_metrics
    from src.report_generator import generate_pdf_report
    rec("PASS", "1. Imports")
except Exception:
    traceback.print_exc()
    rec("FAIL", "1. Imports", "fix the error above first")
    sys.exit(1)

# ------------------------------------------------------ training record
if not os.path.exists(CKPT):
    rec("FAIL", "2. Checkpoint file", f"missing: {CKPT}")
else:
    rec("PASS", "2. Checkpoint file", f"{os.path.getsize(CKPT) / 1e6:.1f} MB, modified {time.ctime(os.path.getmtime(CKPT))}")

hist = []
if not os.path.exists(HIST):
    rec("FAIL", "3. training_history.json", "missing - Tab 6 has nothing real to show")
else:
    hist = json.load(open(HIST, encoding="utf-8"))
    best = max(hist, key=lambda r: r["dice"]["mean"])
    detail = f"{len(hist)} epochs, best epoch {best['epoch']}, mean Dice {best['dice']['mean']}, val cases {best.get('val_cases', '?')}, history file modified {time.ctime(os.path.getmtime(HIST))}"
    if len(hist) < 10:
        rec("WARN", "3. Training length", detail + " (very few epochs)")
    else:
        rec("PASS", "3. Training record", detail)
    if not best.get("held_out", False):
        rec("WARN", "3b. Held-out split", "history says validation overlapped training; scores are NOT held-out")
    else:
        rec("PASS", "3b. Held-out split", "history says validation used held-out patients")

# ----------------------------------------------------------- load model
engine = None
try:
    engine = BrainTumorInferenceEngine(checkpoint_path=CKPT)
    if engine.checkpoint_loaded:
        rec("PASS", "4. Trained weights load", f"device = {engine.device}")
    else:
        rec("FAIL", "4. Trained weights load", "checkpoint did NOT load - model is untrained or architecture changed")
except Exception:
    traceback.print_exc()
    rec("FAIL", "4. Model construction")

# --------------------------------------------------- held-out evaluation
data_dir = os.path.join(ROOT, "data", "BraTS2021_Training_Data")
files = load_brats_file_list(data_dir)
if not files:
    data_dir = os.path.join(ROOT, "data", "BraTS2021_Sample")
    files = load_brats_file_list(data_dir)
files = [f for f in files if "label" in f]

if not files:
    rec("FAIL", "5. Data", "no cases with ground-truth labels found under data/")
elif engine is not None:
    if len(files) >= 5:
        sh = list(files)
        random.Random(42).shuffle(sh)  # same split rule as train.py
        val_files = sh[: max(1, int(0.2 * len(sh)))]
        held = True
    else:
        val_files, held = files, False
        rec("WARN", "5. Data", f"only {len(files)} labelled case(s): evaluation cannot be held-out")
    val_files = val_files[: args.max_cases]
    print(f"\nData folder: {data_dir} ({len(files)} labelled case(s) found)")
    print(f"Evaluating {len(val_files)} case(s): {[c['patient_id'] for c in val_files]}\n")

    per_case, last_pred, last_pid = [], None, None
    tf = get_val_transforms()
    for c in val_files:
        pid = c["patient_id"]
        try:
            d = tf({"image": c["image"], "label": c["label"]})
            tgt = convert_labels_to_brats_classes(d["label"]).numpy()
            res = engine.predict_full_volume(d["image"].float(), overlap=0.25)
            pred = res["binary_3channel"]
            m = compute_clinical_validation_metrics(pred, tgt)
            wt_p, wt_t = int(pred[0].sum()), int(tgt[0].sum())
            print(f"  {pid}: Dice WT {m['Whole Tumor (WT)']['Dice_Similarity_Coefficient']:.3f} | "
                  f"TC {m['Tumor Core (TC)']['Dice_Similarity_Coefficient']:.3f} | "
                  f"ET {m['Enhancing Tumor (ET)']['Dice_Similarity_Coefficient']:.3f} | "
                  f"WT HD95 {m['Whole Tumor (WT)']['HD95_mm']} mm | pred/true WT voxels {wt_p}/{wt_t}")
            probs = res["probabilities"][0]
            inside = probs[tgt[0] > 0.5]
            outside = probs[tgt[0] <= 0.5]
            in_mean = float(inside.mean()) if inside.size else float("nan")
            print(f"      diagnostics: volume shape {tuple(d['image'].shape)}, "
                  f"label values {sorted(np.unique(d['label'].numpy()).astype(int).tolist())}, "
                  f"WT prob inside true tumor mean {in_mean:.3f}, outside mean {float(outside.mean()):.3f}, max {float(probs.max()):.3f}")
            wt_dice = m["Whole Tumor (WT)"]["Dice_Similarity_Coefficient"]
            if wt_p == 0 or wt_dice < 0.5:
                rec("FAIL", f"6. {pid} segmentation quality", f"WT Dice {wt_dice:.3f}, predicted {wt_p} vs true {wt_t} WT voxels")
            elif wt_t and wt_p > 5 * wt_t:
                rec("WARN", f"6. {pid} prediction", "predicted WT is >5x the true size")
            per_case.append((pid, m))
            last_pred, last_pid = pred, pid

            # ---- app path: raw volume + the app's own normalization, no orientation transform
            raw = np.stack([nib.load(p).get_fdata() for p in c["image"]]).astype(np.float32)
            for ch in range(raw.shape[0]):
                mk = raw[ch] != 0
                if mk.any():
                    raw[ch][mk] = (raw[ch][mk] - raw[ch][mk].mean()) / (raw[ch][mk].std() + 1e-8)
            seg_raw = nib.load(c["label"]).get_fdata()
            tgt_raw = convert_labels_to_brats_classes(torch.from_numpy(seg_raw).unsqueeze(0)).numpy()
            res_app = engine.predict_full_volume(raw, overlap=0.25)
            m_app = compute_clinical_validation_metrics(res_app["binary_3channel"], tgt_raw)
            a, b = m_app["Whole Tumor (WT)"]["Dice_Similarity_Coefficient"], m["Whole Tumor (WT)"]["Dice_Similarity_Coefficient"]
            if max(a, b) < 0.05:
                rec("WARN", f"7. {pid} app path vs training path", "both ~0, so this comparison says nothing until the model finds the tumor")
            elif a < b - 0.10:
                rec("FAIL", f"7. {pid} app path matches training path",
                    f"WT Dice {a:.3f} (app) vs {b:.3f} (training pipeline): likely orientation mismatch, send me this output")
            else:
                rec("PASS", f"7. {pid} app path matches training path", f"WT Dice {a:.3f} (app) vs {b:.3f} (training pipeline)")
        except Exception:
            traceback.print_exc()
            rec("FAIL", f"6. {pid} evaluation", "see error above")

    if per_case:
        def avg(region, key):
            v = [m[region][key] for _, m in per_case if m[region][key] == m[region][key]]
            return sum(v) / len(v) if v else float("nan")
        wt_d = avg("Whole Tumor (WT)", "Dice_Similarity_Coefficient")
        et_d = avg("Enhancing Tumor (ET)", "Dice_Similarity_Coefficient")
        wt_h = avg("Whole Tumor (WT)", "HD95_mm")
        print("\n" + "-" * 72)
        print(f" TARGETS  (mean over {len(per_case)} case(s); held-out split: {held})")
        print("-" * 72)
        print(f"  WT Dice  {wt_d:.3f}  (target >= {DICE_T['WT']})  -> {'MET' if wt_d >= DICE_T['WT'] else 'NOT MET'}")
        print(f"  ET Dice  {et_d:.3f}  (target >= {DICE_T['ET']})  -> {'MET' if et_d >= DICE_T['ET'] else 'NOT MET'}")
        print(f"  WT HD95  {wt_h:.2f} mm (target <= {HD95_T})  -> {'MET' if wt_h <= HD95_T else 'NOT MET'}")
        print("  These are your real numbers. Report them as they are.\n")
        if hist:
            hb = max(hist, key=lambda r: r["dice"]["mean"])["dice"]["mean"]
            ev = (wt_d + avg("Tumor Core (TC)", "Dice_Similarity_Coefficient") + et_d) / 3
            if hb > 0.8 and ev < 0.5:
                rec("FAIL", "10. Training record matches this checkpoint",
                    f"history says best mean Dice {hb:.3f} but evaluating now gives {ev:.3f}: the record, checkpoint or data disagree")

    # ------------------------------------------- volumetrics, RANO, PDF
    if last_pred is not None:
        try:
            vm = compute_subregion_volumetrics(last_pred)
            r1 = compute_rano_criteria_metrics(vm["cleaned_masks"][0])
            r2 = compute_rano_criteria_metrics(vm["cleaned_masks"][0])
            rec("PASS" if r1 == r2 else "FAIL", "8. RANO is repeatable", f"D1 {r1['max_diameter_d1_mm']} mm both runs" if r1 == r2 else "values changed between runs")
            pdf = generate_pdf_report(last_pid, vm, None, mask_source="SegResNet model prediction")
            if not pdf.startswith(b"%PDF"):
                rec("FAIL", "9. PDF report", "output is not a PDF")
            else:
                try:
                    import io
                    # pyrefly: ignore[missing-import]
                    from pypdf import PdfReader
                    txt = " ".join(pg.extract_text() or "" for pg in PdfReader(io.BytesIO(pdf)).pages)
                    if "Passed (DSC" in txt:
                        rec("FAIL", "9. PDF report", "still contains the hardcoded 'Passed' validation claim")
                    elif "not clinically validated" in txt:
                        rec("PASS", "9. PDF report", f"{len(pdf) / 1024:.0f} KB, shows research-prototype status")
                    else:
                        rec("WARN", "9. PDF report", "generated, but could not find the research-prototype wording; open it and check")
                except ImportError:
                    rec("WARN", "9. PDF report", "generated OK; run 'pip install pypdf' to auto-check its text, or open it manually")
        except Exception:
            traceback.print_exc()
            rec("FAIL", "8-9. Volumetrics / PDF")

print("=" * 72)
print(f" Summary: {results.count('PASS')} PASS | {results.count('WARN')} WARN | {results.count('FAIL')} FAIL")
print(" PASS means the check ran correctly. It does not mean the targets were met (see TARGETS).")
if results.count("FAIL"):
    print(" There are FAILs: do not submit these numbers or this model until they are resolved.")
print("=" * 72)
