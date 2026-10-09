#!/usr/bin/env python
# coding: utf-8

## fine-tune the CERTH blur classifier (cnn_blur_classification.py) on the chili dataset and test it
## usage: python finetune_chili.py [--arch resnet34|resnet50] [--size 448] [--init certh|imagenet]
##
## chili layout: <data>/{good,bad}/{fwd,back,overhead}/<id>_<clip>_frameNN.jpg   (bad = blurred)
## frames with the same <id> come from one event, so the split is done by id (no leakage between splits)

import argparse
import warnings
warnings.filterwarnings('ignore')

import cv2
import pandas as pd
from fastai.vision.all import *
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

parser = argparse.ArgumentParser()
parser.add_argument("--arch", default="resnet34", choices=["resnet34", "resnet50"])
parser.add_argument("--data", default="~/p1/data/chili")
parser.add_argument("--certh", default="./CERTH_ImageBlurDataset")  ## where cnn_blur_classification.py saved its models
parser.add_argument("--init", default="certh", choices=["certh", "imagenet"])
parser.add_argument("--size", type=int, default=448)  ## 224 loses too much of the blur signal on these 2-5MP frames
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--epochs", type=int, nargs=2, default=[4, 8], help="frozen / unfrozen epochs")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

root = Path(args.data).expanduser()
classes = ["bad", "good"]
tag = f"{args.arch}_{args.init}_{args.size}"
certh_ckpt = Path(args.certh)/"TrainingSet"/"models"/f"model_1_stage_4_{args.arch}.pth"

print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")


## data : 60/20/20 train/valid/test split, grouped by event id and stratified by label
files = sorted(get_image_files(root))
df = pd.DataFrame({"path": [str(f) for f in files], "label": [f.parent.parent.name for f in files],
                   "camera": [f.parent.name for f in files], "id": [f.name.split("_")[0] for f in files]})
folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed).split(df, df["label"], df["id"])
folds = [f[1] for f in folds]
df["split"] = "train"
df.loc[folds[0], "split"] = "test"
df.loc[folds[1], "split"] = "valid"
print(pd.crosstab(df["split"], df["label"], margins=True))
test = df[df["split"] == "test"].reset_index(drop=True)
valid = df[df["split"] == "valid"].reset_index(drop=True)
trainval = df[df["split"] != "test"].reset_index(drop=True)
trainval["is_valid"] = trainval["split"] == "valid"
y_test, y_valid = (test["label"] == "bad").values, (valid["label"] == "bad").values

set_seed(args.seed, reproducible=True)
data = ImageDataLoaders.from_df(trainval, path="/", fn_col="path", label_col="label", valid_col="is_valid",
        item_tfms=Resize(args.size), batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)],
        bs=args.bs, num_workers=8)
assert list(data.vocab) == classes


## bad (blurred) is the positive class
def report(name, y_true, p_bad, thresh=0.5):
    y_pred = p_bad >= thresh
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    res = {"model": name, "thresh": thresh, "accuracy": accuracy_score(y_true, y_pred), "precision": prec,
           "recall": rec, "f1": f1, "roc_auc": roc_auc_score(y_true, p_bad), "avg_prec": average_precision_score(y_true, p_bad)}
    print(f"\n{name} (thresh {thresh:.3g})")
    print("  " + "  ".join(f"{k} {v:.4f}" for k, v in list(res.items())[2:]))
    print("  confusion matrix (rows = actual bad/good, cols = predicted bad/good):")
    print("  " + str(confusion_matrix(~y_true, ~y_pred)).replace("\n", "\n  "))
    return res

def best_f1_thresh(y_true, score):
    cands = np.unique(score)
    return float(cands[np.argmax([precision_recall_fscore_support(y_true, score >= t, average="binary", zero_division=0)[2] for t in cands])])

results = []
results.append(report("always good (majority class)", y_test, np.zeros(len(test))))


## baseline 1 : Laplacian variance, threshold picked on train+valid (low variance -> blurred)
def lap_var(fn): return cv2.Laplacian(cv2.imread(fn, cv2.IMREAD_GRAYSCALE), cv2.CV_64F).var()
df["lap_var"] = parallel(lap_var, df["path"], n_workers=16, progress=False)
tv = df[df["split"] != "test"]
lap_thresh = best_f1_thresh((tv["label"] == "bad").values, -tv["lap_var"].values)
lap_test = -df.loc[df["split"] == "test", "lap_var"].values
print(f"\nLaplacian variance threshold picked on train+valid: {-lap_thresh:.2f}")
results.append(report("Laplacian variance", y_test, lap_test, lap_thresh))


## baseline 2 : the CERTH model as is. classes 0-1 (artificially / naturally blurred) -> bad, 2 (undistorted) -> good
def p_bad(learn, frame, n_bad):
    probs, _ = learn.get_preds(dl=learn.dls.test_dl(frame["path"].tolist()))
    return probs[:, :n_bad].sum(dim=1).numpy()

certh = vision_learner(data, getattr(models, args.arch), n_out=3, path=".")
certh.model.load_state_dict(torch.load(certh_ckpt, map_location="cpu", weights_only=False)["model"])
results.append(report("CERTH model, no fine-tuning", y_test, p_bad(certh, test, 2)))
del certh


## fine-tune : new 2 class head, class weighted loss (bad is ~10% of the data)
counts = trainval[~trainval["is_valid"]]["label"].value_counts()[classes].values
weights = tensor(counts.sum() / (len(classes) * counts)).float()
learn = vision_learner(data, getattr(models, args.arch), metrics=[error_rate, accuracy], path=".",
                       loss_func=CrossEntropyLossFlat(weight=weights.to(data.device)))
if args.init == "certh":
    state, own = torch.load(certh_ckpt, map_location="cpu", weights_only=False)["model"], learn.model.state_dict()
    state = {k: v for k, v in state.items() if v.shape == own[k].shape}  ## everything but the final 3 class layer
    print(f"\nloaded {len(state)}/{len(own)} tensors from {certh_ckpt}")
    learn.model.load_state_dict(state, strict=False)

learn.fit_one_cycle(args.epochs[0])
learn.unfreeze()
learn.fit_one_cycle(args.epochs[1], lr_max=slice(3e-5, 3e-4))
learn.save(f"chili_{tag}")
print("saved", learn.path/learn.model_dir/f"chili_{tag}.pth")


## test : at 0.5 and at the threshold that maximises F1 on the validation split
p_valid, p_test = p_bad(learn, valid, 1), p_bad(learn, test, 1)
report("fine-tuned, validation split", y_valid, p_valid)
results.append(report(f"fine-tuned {tag}", y_test, p_test))
results.append(report(f"fine-tuned {tag}", y_test, p_test, best_f1_thresh(y_valid, p_valid)))

test["p_bad"] = p_test
print("\ntest accuracy by camera (thresh 0.5):")
print(test.groupby("camera").apply(lambda g: pd.Series({"n": len(g), "n_bad": (g["label"] == "bad").sum(),
        "accuracy": ((g["p_bad"] >= 0.5) == (g["label"] == "bad")).mean()})))

print("\ntest summary:")
print(pd.DataFrame(results).round(4).to_string(index=False))
pd.DataFrame(results).to_csv(f"chili_results_{tag}.csv", index=False)
test.to_csv(f"chili_preds_{tag}.csv", index=False)  ## store the results
