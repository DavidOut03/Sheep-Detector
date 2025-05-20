
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
from ultralytics import YOLO
import cv2


ROOT_FOLDER = "."
DATASET_FOLDER = f"{ROOT_FOLDER}/dataset"
IMAGE_TEST_FOLDER = f"{DATASET_FOLDER}/images/test"
LABEL_TEST_FOLDER = f"{DATASET_FOLDER}/labels/test"

model = YOLO(f"{ROOT_FOLDER}/model/sheep-detector+preprocessing+augmentation.pt")

def load_yolo_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, x_center, y_center, width, height = map(float, parts)
                boxes.append((x_center, y_center, width, height))
    return boxes

def yolo_to_xyxy(box, img_w, img_h):
    x_c, y_c, w, h = box
    x1 = round((x_c - w / 2) * img_w)
    y1 = round((y_c - h / 2) * img_h)
    x2 = round((x_c + w / 2) * img_w)
    y2 = round((y_c + h / 2) * img_h)
    return x1, y1, x2, y2

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) else 0

class ImageViewer:
    def __init__(self, root, image_paths):
        self.root = root
        self.image_paths = image_paths
        self.index = 0
        self.zoom = 1.0

        self.root.title("YOLO Viewer")
        self.root.geometry("1000x750")
        self.root.configure(bg="#000")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=6, relief="flat", background="#333", foreground="#fff")
        style.map("TButton", background=[("active", "#555")], foreground=[("active", "#fff")])

        self.left_frame = tk.Frame(self.root, width=230, bg="#000")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.stats_labels = {
            "precision": tk.Label(self.left_frame, text="", fg="white", bg="#000", anchor="w"),
            "recall": tk.Label(self.left_frame, text="", fg="white", bg="#000", anchor="w"),
            "f1": tk.Label(self.left_frame, text="", fg="white", bg="#000", anchor="w"),
            "iou": tk.Label(self.left_frame, text="", fg="white", bg="#000", anchor="w"),
            "found": tk.Label(self.left_frame, text="", fg="white", bg="#000", anchor="w"),
            "actual": tk.Label(self.left_frame, text="", fg="white", bg="#000", anchor="w"),
            "legend1": tk.Label(self.left_frame, text="🟥 Prediction", fg="red", bg="#000", anchor="w"),
            "legend2": tk.Label(self.left_frame, text="🟩 Ground Truth", fg="green", bg="#000", anchor="w"),
        }

        for label in self.stats_labels.values():
            label.pack(pady=3, padx=10, anchor="w")

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", self.zoom_image)

        self.bottom_frame = tk.Frame(self.root, bg="#000")
        self.bottom_frame.pack(side=tk.BOTTOM, pady=12)
        ttk.Button(self.bottom_frame, text="Previous", command=self.show_prev).pack(side=tk.LEFT, padx=20)
        ttk.Button(self.bottom_frame, text="Next", command=self.show_next).pack(side=tk.RIGHT, padx=20)

        self.root.bind("<Configure>", self.redraw_image)
        self.load_image(self.index)

    def load_image(self, idx):
        image_path = self.image_paths[idx]
        label_path = os.path.join(LABEL_TEST_FOLDER, os.path.splitext(os.path.basename(image_path))[0] + ".txt")

        img_cv = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        self.original_image = Image.fromarray(img_rgb)
        img_w, img_h = self.original_image.size

        display_img = self.original_image.copy()
        draw = ImageDraw.Draw(display_img)

        gt_boxes = [yolo_to_xyxy(b, img_w, img_h) for b in load_yolo_labels(label_path)]
        for box in gt_boxes:
            draw.rectangle(box, outline="green", width=2)

        predictions = model(image_path)[0]
        pred_boxes = [tuple(map(int, box[:4])) for box in predictions.boxes.xyxy.cpu().numpy()]
        for box in pred_boxes:
            draw.rectangle(box, outline="red", width=2)

        iou_list = []
        matched_gt = set()
        tp = 0

        for p_box in pred_boxes:
            best_iou = 0
            best_idx = -1
            for i, g_box in enumerate(gt_boxes):
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= 0.5 and best_idx not in matched_gt:
                matched_gt.add(best_idx)
                tp += 1
                iou_list.append(best_iou)

        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        avg_iou = sum(iou_list) / len(iou_list) if iou_list else 0

        self.stats_labels["precision"].config(text=f"Precision: {precision:.2f}")
        self.stats_labels["recall"].config(text=f"Recall: {recall:.2f}")
        self.stats_labels["f1"].config(text=f"F1 Score: {f1:.2f}")
        self.stats_labels["iou"].config(text=f"IoU (avg): {avg_iou:.2f}")
        self.stats_labels["found"].config(text=f"Found sheep: {len(pred_boxes)}")
        self.stats_labels["actual"].config(text=f"Actual sheep: {len(gt_boxes)}")

        self.display_image = display_img
        self.zoom = 1.0
        self.update_image()

    def update_image(self):
        if self.display_image:
            w, h = self.display_image.size
            zoomed = self.display_image.resize((int(w * self.zoom), int(h * self.zoom)), Image.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(zoomed)
            self.redraw_image()

    def redraw_image(self, event=None):
        self.canvas.delete("all")
        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()
        i_w = self.tk_img.width()
        i_h = self.tk_img.height()
        x = (c_w - i_w) // 2
        y = (c_h - i_h) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_img)

    def zoom_image(self, event):
        delta = 0.1 if event.delta > 0 else -0.1
        self.zoom = max(0.2, min(5.0, self.zoom + delta))
        self.update_image()

    def show_next(self):
        if self.index < len(self.image_paths) - 1:
            self.index += 1
            self.load_image(self.index)

    def show_prev(self):
        if self.index > 0:
            self.index -= 1
            self.load_image(self.index)

    def on_close(self):
        self.root.destroy()
        exit()

def get_image_paths(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

if __name__ == "__main__":
    image_files = get_image_paths(IMAGE_TEST_FOLDER)
    if not image_files:
        print("No images found.")
    else:
        root = tk.Tk()
        app = ImageViewer(root, image_files)
        root.mainloop()
