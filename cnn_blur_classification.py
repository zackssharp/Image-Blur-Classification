#!/usr/bin/env python
# coding: utf-8

## fastai v2 port of "CNN (ResNet 50) Blur_Image_Classification.ipynb"
## usage: python cnn_blur_classification.py [--arch resnet34|resnet50]

import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from fastai.vision.all import *
from sklearn.metrics import accuracy_score, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--arch", default="resnet34", choices=["resnet34", "resnet50"])  ## notebook code uses resnet34
parser.add_argument("--data", default="./CERTH_ImageBlurDataset")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

root = Path(args.data)
path = root/"TrainingSet"
classes = ["Artificially-Blurred", "Naturally-Blurred", "Undistorted"]

print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")


## data : 80/20 random split, 224px, default augmentations, imagenet normalization
set_seed(args.seed, reproducible=True)
data = ImageDataLoaders.from_folder(path, train=".", valid_pct=0.2, seed=args.seed,
        item_tfms=Resize(224), batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)],
        num_workers=8)
print(data.vocab, data.c, len(data.train_ds), len(data.valid_ds))
assert list(data.vocab) == classes


## train model
learn = vision_learner(data, getattr(models, args.arch), metrics=[error_rate, accuracy])
learn.fit_one_cycle(4)
learn.save(f"model_1_stage_1_{args.arch}")

learn.unfreeze()
learn.fit_one_cycle(2, lr_max=slice(3e-5, 3e-4))
learn.save(f"model_1_stage_4_{args.arch}")


## evaluating on the validation split (the notebook's confusion matrix)
interp = ClassificationInterpretation.from_learner(learn)
cm = interp.confusion_matrix()
print("validation confusion matrix (rows = actual, cols = predicted):", classes)
print(cm)
print("validation accuracy (3 classes):", cm.trace() / cm.sum())
blur_cm = np.array([[cm[:2, :2].sum(), cm[:2, 2].sum()], [cm[2, :2].sum(), cm[2, 2]]])
print("validation accuracy (blurred vs undistorted):", blur_cm.trace() / blur_cm.sum())


## testing for naturalBlurSet and digitalBlurSet
## tensor 0-1 -> blurred (1) , tensor 2 -> undistorted (-1)
def evaluate(set_name, names, labels):
    files = [root/"EvaluationSet"/set_name/n for n in names]
    probs, _ = learn.get_preds(dl=learn.dls.test_dl(files))
    idx = probs.argmax(dim=1).numpy()
    y_pred = np.where(idx == 2, -1, 1)
    print(f"{set_name} accuracy:", accuracy_score(labels, y_pred))
    print(confusion_matrix(labels, y_pred))
    return pd.DataFrame({"set": set_name, "image": names, "predictions": [classes[i] for i in idx],
                         "blur_pred": y_pred, "blur_label": labels})

digital = pd.read_excel(root/"EvaluationSet"/"DigitalBlurSet.xlsx")
natural = pd.read_excel(root/"EvaluationSet"/"NaturalBlurSet.xlsx")
preds_df = pd.concat([
    evaluate("DigitalBlurSet", [n.strip() for n in digital["MyDigital Blur"]], digital.iloc[:, 1].values),
    evaluate("NaturalBlurSet", [n.strip() + ".jpg" for n in natural["Image Name"]], natural["Blur Label"].values),
])
preds_df.to_csv(f"preds_{args.arch}.csv", index=False)  ## store the results
