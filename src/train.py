from pathlib import Path
import argparse
import csv
import shutil
from typing import Dict, List, Tuple

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from ultralytics import YOLO


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def get_class_names(data_yaml_path: Path) -> List[str]:
    data_cfg = load_yaml(data_yaml_path)
    names = data_cfg["names"]

    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]

    return list(names)


def resolve_split_paths(data_yaml_path: Path, split: str) -> Tuple[Path, Path]:
    data_cfg = load_yaml(data_yaml_path)
    dataset_root = data_yaml_path.parent

    split_key = "val" if split in ["val", "valid"] else split

    if split_key not in data_cfg:
        raise ValueError(f"Split '{split_key}' not found in {data_yaml_path}")

    image_dir = Path(data_cfg[split_key])

    if not image_dir.is_absolute():
        image_dir = (dataset_root / image_dir).resolve()

    label_dir = Path(str(image_dir).replace("images", "labels"))

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    return image_dir, label_dir


def read_yolo_labels(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, np.ndarray]]:
    if not label_path.exists():
        return []

    items = []

    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        cls_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])

        x1 = (x - w / 2) * img_w
        y1 = (y - h / 2) * img_h
        x2 = (x + w / 2) * img_w
        y2 = (y + h / 2) * img_h

        items.append((cls_id, np.array([x1, y1, x2, y2], dtype=np.float32)))

    return items


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def build_custom_confusion_matrix(
    model_path: Path,
    data_yaml_path: Path,
    split: str,
    class_names: List[str],
    output_dir: Path,
    conf: float = 0.25,
    iou_thr: float = 0.50,
) -> None:
    model = YOLO(str(model_path))

    image_dir, label_dir = resolve_split_paths(data_yaml_path, split)
    image_paths = []

    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(image_dir.glob(ext))

    image_paths = sorted(image_paths)

    n_classes = len(class_names)
    bg_row = n_classes
    bg_col = n_classes

    matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)

    for image_path in image_paths:
        with Image.open(image_path) as img:
            img_w, img_h = img.size

        label_path = label_dir / f"{image_path.stem}.txt"
        gt_items = read_yolo_labels(label_path, img_w, img_h)

        result = model.predict(
            source=str(image_path),
            conf=conf,
            iou=0.7,
            verbose=False,
        )[0]

        pred_items = []

        if result.boxes is not None and len(result.boxes) > 0:
            pred_boxes = result.boxes.xyxy.cpu().numpy()
            pred_cls = result.boxes.cls.cpu().numpy().astype(int)

            for cls_id, box in zip(pred_cls, pred_boxes):
                pred_items.append((cls_id, box.astype(np.float32)))

        pairs = []

        for gi, (_, gt_box) in enumerate(gt_items):
            for pi, (_, pred_box) in enumerate(pred_items):
                iou = box_iou(gt_box, pred_box)
                if iou >= iou_thr:
                    pairs.append((iou, gi, pi))

        pairs.sort(reverse=True, key=lambda x: x[0])

        matched_g = set()
        matched_p = set()

        for _, gi, pi in pairs:
            if gi in matched_g or pi in matched_p:
                continue

            gt_cls, _ = gt_items[gi]
            pred_cls, _ = pred_items[pi]

            matrix[gt_cls, pred_cls] += 1

            matched_g.add(gi)
            matched_p.add(pi)

        for gi, (gt_cls, _) in enumerate(gt_items):
            if gi not in matched_g:
                matrix[gt_cls, bg_col] += 1

        for pi, (pred_cls, _) in enumerate(pred_items):
            if pi not in matched_p:
                matrix[bg_row, pred_cls] += 1

    labels = class_names + ["background"]

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"custom_confusion_matrix_{split}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gt/pred"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + row.tolist())

    plot_confusion_matrix(
        matrix=matrix,
        labels=labels,
        save_path=output_dir / f"custom_confusion_matrix_{split}.png",
        title=f"Custom Confusion Matrix ({split})",
    )


def plot_confusion_matrix(matrix: np.ndarray, labels: List[str], save_path: Path, title: str) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "white_to_navy",
        ["#ffffff", "#dbeafe", "#60a5fa", "#1e3a8a", "#020617"],
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap=cmap)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)

    max_value = matrix.max() if matrix.size else 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if max_value > 0 and value > max_value * 0.55 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def clean_results_csv(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    return df


def find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def plot_line(df: pd.DataFrame, columns: List[str], labels: List[str], title: str, ylabel: str, save_path: Path) -> None:
    existing = [(c, l) for c, l in zip(columns, labels) if c in df.columns]

    if not existing:
        return

    x_col = "epoch" if "epoch" in df.columns else None
    x = df[x_col] if x_col else np.arange(len(df))

    fig, ax = plt.subplots(figsize=(9, 5))

    for col, label in existing:
        ax.plot(x, df[col], label=label, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def create_training_plots(run_dir: Path) -> None:
    results_csv = run_dir / "results.csv"

    if not results_csv.exists():
        print(f"results.csv not found at {results_csv}")
        return

    df = clean_results_csv(results_csv)

    custom_dir = run_dir / "custom_plots"
    custom_dir.mkdir(parents=True, exist_ok=True)

    plot_line(
        df,
        ["train/box_loss", "val/box_loss"],
        ["train box loss", "val box loss"],
        "Box Loss",
        "Loss",
        custom_dir / "box_loss.png",
    )

    plot_line(
        df,
        ["train/cls_loss", "val/cls_loss"],
        ["train cls loss", "val cls loss"],
        "Classification Loss",
        "Loss",
        custom_dir / "cls_loss.png",
    )

    plot_line(
        df,
        ["train/dfl_loss", "val/dfl_loss"],
        ["train dfl loss", "val dfl loss"],
        "DFL Loss",
        "Loss",
        custom_dir / "dfl_loss.png",
    )

    plot_line(
        df,
        ["metrics/precision(B)", "metrics/recall(B)"],
        ["precision", "recall"],
        "Precision and Recall",
        "Score",
        custom_dir / "precision_recall.png",
    )

    plot_line(
        df,
        ["metrics/mAP50(B)", "metrics/mAP50-95(B)"],
        ["mAP@50", "mAP@50:95"],
        "mAP Metrics",
        "mAP",
        custom_dir / "map_metrics.png",
    )

    precision_col = find_col(df, ["metrics/precision(B)"])
    recall_col = find_col(df, ["metrics/recall(B)"])

    if precision_col and recall_col:
        precision = df[precision_col].astype(float)
        recall = df[recall_col].astype(float)
        df["custom/F1"] = (2 * precision * recall) / (precision + recall + 1e-9)

        plot_line(
            df,
            ["custom/F1"],
            ["F1"],
            "F1 Score",
            "F1",
            custom_dir / "f1_score.png",
        )


def create_metric_summary(run_dir: Path) -> None:
    results_csv = run_dir / "results.csv"

    if not results_csv.exists():
        return

    df = clean_results_csv(results_csv)

    map_col = find_col(df, ["metrics/mAP50-95(B)"])
    if map_col:
        best_idx = df[map_col].astype(float).idxmax()
    else:
        best_idx = len(df) - 1

    best = df.loc[best_idx]

    summary = {
        "best_epoch": int(best["epoch"]) if "epoch" in df.columns else int(best_idx),
        "precision": float(best["metrics/precision(B)"]) if "metrics/precision(B)" in df.columns else None,
        "recall": float(best["metrics/recall(B)"]) if "metrics/recall(B)" in df.columns else None,
        "mAP50": float(best["metrics/mAP50(B)"]) if "metrics/mAP50(B)" in df.columns else None,
        "mAP50_95": float(best["metrics/mAP50-95(B)"]) if "metrics/mAP50-95(B)" in df.columns else None,
        "train_box_loss": float(best["train/box_loss"]) if "train/box_loss" in df.columns else None,
        "val_box_loss": float(best["val/box_loss"]) if "val/box_loss" in df.columns else None,
        "train_cls_loss": float(best["train/cls_loss"]) if "train/cls_loss" in df.columns else None,
        "val_cls_loss": float(best["val/cls_loss"]) if "val/cls_loss" in df.columns else None,
    }

    precision = summary["precision"]
    recall = summary["recall"]

    if precision is not None and recall is not None:
        summary["F1"] = float((2 * precision * recall) / (precision + recall + 1e-9))
    else:
        summary["F1"] = None

    pd.DataFrame([summary]).to_csv(run_dir / "metrics_summary.csv", index=False)

    with open(run_dir / "metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write("YOLO Baseline Training Summary\n")
        f.write("=" * 40 + "\n")

        for key, value in summary.items():
            f.write(f"{key}: {value}\n")


def copy_config_to_run(config_path: Path, run_dir: Path) -> None:
    shutil.copy2(config_path, run_dir / "experiment_config.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--data", default=None, help="Override dataset data.yaml path")
    parser.add_argument("--project", default=None, help="Override output project directory")
    parser.add_argument("--device", default=None, help="Override device, e.g. 0 or cpu")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_yaml(config_path)

    if args.data:
        cfg["data"] = args.data

    if args.project:
        cfg["project"] = args.project

    if args.device:
        cfg["device"] = args.device

    model = YOLO(cfg["model"])

    model.train(
        data=cfg["data"],
        epochs=cfg.get("epochs", 50),
        imgsz=cfg.get("imgsz", 640),
        batch=cfg.get("batch", 16),
        device=cfg.get("device", 0),
        workers=cfg.get("workers", 2),
        seed=cfg.get("seed", 42),
        patience=cfg.get("patience", 15),
        project=cfg.get("project", "runs"),
        name=cfg.get("name", cfg.get("experiment_name", "exp")),
        exist_ok=cfg.get("exist_ok", True),
        plots=cfg.get("plots", True),
    )

    run_dir = Path(model.trainer.save_dir)
    print(f"Run directory: {run_dir}")

    copy_config_to_run(config_path, run_dir)
    create_metric_summary(run_dir)
    create_training_plots(run_dir)

    best_model_path = run_dir / "weights" / "best.pt"
    data_yaml_path = Path(cfg["data"]).resolve()

    if best_model_path.exists() and data_yaml_path.exists():
        class_names = get_class_names(data_yaml_path)

        build_custom_confusion_matrix(
            model_path=best_model_path,
            data_yaml_path=data_yaml_path,
            split=cfg.get("custom_eval_split", "test"),
            class_names=class_names,
            output_dir=run_dir / "custom_plots",
            conf=cfg.get("custom_conf", 0.25),
            iou_thr=cfg.get("custom_iou", 0.50),
        )

    print("Training and custom logging finished.")


if __name__ == "__main__":
    main()