#!/usr/bin/env python3
"""
Run Deformable DETR predictions on a small set of images.

Outputs:
  - predictions.json with boxes, labels, scores, and uncertainty values
  - annotated PNG/JPG copies with predicted boxes drawn on top
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent
DEFORMABLE_DETR_ROOT = PROJECT_ROOT / "Deformable-DETR"
sys.path.insert(0, str(DEFORMABLE_DETR_ROOT))

from datasets.coco import make_coco_transforms  # noqa: E402
from main import get_args_parser  # noqa: E402
from models import build_model  # noqa: E402
from util.uncertainty import mc_dropout_forward  # noqa: E402


COCO_CATEGORY_NAMES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}


def collect_images(inputs, input_dir):
    image_paths = []
    if inputs:
        image_paths.extend(Path(p) for p in inputs)
    if input_dir:
        root = Path(input_dir)
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            image_paths.extend(sorted(root.glob(pattern)))
    image_paths = [p for p in image_paths if p.exists()]
    if not image_paths:
        raise SystemExit("No input images found.")
    return image_paths


def tensor_to_float_list(tensor):
    return [float(x) for x in tensor.detach().cpu().tolist()]


def draw_predictions(image, detections, output_path):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        x0, y0, x1, y1 = det["box"]
        label = f'{det["label_name"]} {det["score"]:.2f} U={det["uncertainty"]:.2f}'
        uncertainty = max(0.0, min(float(det["uncertainty"]), 1.0))
        color = (int(80 + 175 * uncertainty), int(210 * (1.0 - uncertainty)), 64)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        text_box = draw.textbbox((x0, y0), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        y_text = max(0, y0 - text_h - 4)
        draw.rectangle([x0, y_text, x0 + text_w + 6, y_text + text_h + 4], fill=color)
        draw.text((x0 + 3, y_text + 2), label, fill=(255, 255, 255), font=font)

    canvas.save(output_path)


def run_prediction(args):
    device = torch.device(args.device)
    model, _, postprocessors = build_model(args)
    model.to(device)
    model.eval()

    if not args.resume:
        raise SystemExit("Please pass --resume path/to/checkpoint.pth")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)

    transform = make_coco_transforms("val")
    image_paths = collect_images(args.input, args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_outputs = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        sample, _ = transform(image, None)
        sample = sample.to(device)

        with torch.no_grad():
            if args.mc_dropout:
                outputs = mc_dropout_forward(
                    model,
                    [sample],
                    runs=args.mc_runs,
                    scope=args.mc_dropout_scope,
                )
            else:
                outputs = model([sample])

            target_sizes = torch.tensor([[height, width]], dtype=torch.float32, device=device)
            result = postprocessors["bbox"](outputs, target_sizes)[0]

        keep = result["scores"] >= args.score_threshold
        scores = result["scores"][keep][: args.max_detections]
        labels = result["labels"][keep][: args.max_detections]
        boxes = result["boxes"][keep][: args.max_detections]
        uncertainty = result["uncertainty"][keep][: args.max_detections]

        detections = []
        for score, label, box, unc in zip(scores, labels, boxes, uncertainty):
            label_id = int(label.item())
            detections.append({
                "label_id": label_id,
                "label_name": COCO_CATEGORY_NAMES.get(label_id, f"class_{label_id}"),
                "score": float(score.item()),
                "uncertainty": float(unc.item()),
                "box": tensor_to_float_list(box),
            })

        annotated_path = output_dir / f"{image_path.stem}_pred{image_path.suffix}"
        draw_predictions(image, detections, annotated_path)

        all_outputs.append({
            "image": str(image_path),
            "width": width,
            "height": height,
            "annotated_image": str(annotated_path),
            "detections": detections,
        })
        print(f"{image_path}: {len(detections)} detections -> {annotated_path}")

    with (output_dir / "predictions.json").open("w") as f:
        json.dump(all_outputs, f, indent=2)
    print(f"Wrote {output_dir / 'predictions.json'}")


def main():
    parser = argparse.ArgumentParser(
        "Predict on images with Deformable DETR",
        parents=[get_args_parser()],
    )
    parser.add_argument("--input", nargs="*", help="One or more image paths")
    parser.add_argument("--input_dir", help="Directory of images")
    parser.add_argument("--score_threshold", default=0.4, type=float)
    parser.add_argument("--max_detections", default=20, type=int)
    args = parser.parse_args()
    run_prediction(args)


if __name__ == "__main__":
    main()
