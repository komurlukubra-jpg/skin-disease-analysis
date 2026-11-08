# train.py 
import os, json, glob
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

IMG_SIZE = (256, 256)

def list_classes(root="data/train"):
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes = sorted(classes)
    if not classes:
        raise SystemExit("data/train altında sınıf klasörü bulunamadı.")
    return classes

def extract_features(img_path):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img)
    gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.float32) / 255.0
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(16,16),
                   cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)
    hsv = Image.fromarray(arr).convert("HSV")
    hsv = np.array(hsv)
    h, s, v = hsv[...,0], hsv[...,1], hsv[...,2]
    hist, _ = np.histogramdd([h.ravel(), s.ravel(), v.ravel()],
                             bins=(8,8,8), range=((0,255),(0,255),(0,255)))
    color_feat = (hist.flatten() / (hist.sum() + 1e-8)).astype(np.float32)
    return np.concatenate([hog_feat.astype(np.float32), color_feat], axis=0)

def load_split(split_dir, classes):
    X, y = [], []
    for idx, cls in enumerate(classes):
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            for p in glob.glob(os.path.join(split_dir, cls, ext)):
                try:
                    X.append(extract_features(p))
                    y.append(idx)
                except Exception as e:
                    print("skip:", p, e)
    if not X:
        raise SystemExit(f"{split_dir} içinde veri bulunamadı.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def main():
    print(">>> main() başladı")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    classes = list_classes("data/train")
    print("Sınıflar:", classes)

    Xtr, ytr = load_split("data/train", classes)
    Xv,  yv  = load_split("data/val", classes)
    print("Train:", Xtr.shape, "Val:", Xv.shape)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                    probability=True, random_state=42))
    ])
    pipe.fit(Xtr, ytr)

    ypred = pipe.predict(Xv)
    rep = classification_report(yv, ypred, target_names=classes, digits=3)
    print(rep)
    with open("results/report.txt","w",encoding="utf-8") as f:
        f.write(rep)

    cm = confusion_matrix(yv, ypred, labels=list(range(len(classes))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(0.9*len(classes)+3, 0.9*len(classes)+3))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=180)
    plt.close(fig)

    joblib.dump(pipe, "models/model.joblib")
    with open("models/classes.json","w",encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print("Model ve rapor kaydedildi.")

if __name__ == "__main__":
    print(">>> train.py yüklendi")
    main()
