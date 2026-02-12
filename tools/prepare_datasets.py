import os
import zipfile
import random
import shutil
from glob import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET


def convert_neu_annotation(xml_file, target_dir, class_mapping):
    """Converts a single NEU-DET XML annotation to YOLO format."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    img_filename = root.find("filename").text
    img_size = root.find("size")
    img_width = int(img_size.find("width").text)
    img_height = int(img_size.find("height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in class_mapping:
            continue
        class_id = class_mapping[class_name]

        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # Convert to YOLO format
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    if yolo_lines:
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        with open(os.path.join(target_dir, label_filename), "w") as f:
            f.write("\n".join(yolo_lines))

    return img_filename


def convert_deeppcb_annotation(
    txt_file, target_dir, class_mapping, image_width=640, image_height=640
):
    """Converts a single DeepPCB annotation to YOLO format."""
    img_filename = os.path.basename(txt_file).replace(".txt", "_test.jpg")
    yolo_lines = []

    with open(txt_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        # DeepPCB format is xmin ymin xmax ymax class_id
        xmin, ymin, xmax, ymax = map(int, parts[:4])
        original_class_id = int(parts[4])

        # Map original class name
        # DeepPCB classes are 1-6, let's map them to the new combined list
        # This assumes a fixed order for deeppcb_classes
        deeppcb_classes = ["open", "short", "mousebite", "spur", "copper", "pinhole"]
        # The original_class_id from file is 1-based index for deeppcb_classes
        class_name = deeppcb_classes[original_class_id - 1]

        if class_name not in class_mapping:
            continue
        class_id = class_mapping[class_name]

        x_center = (xmin + xmax) / 2 / image_width
        y_center = (ymin + ymax) / 2 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    if yolo_lines:
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        with open(os.path.join(target_dir, label_filename), "w") as f:
            f.write("\n".join(yolo_lines))

    return img_filename


def main():
    # --- Configuration ---
    TEMP_DATA_DIR = "temp_data"
    NEU_REPO_PATH = os.path.join(TEMP_DATA_DIR, "NEU_repo")  # Corrected path
    DEEPPCB_REPO_PATH = os.path.join(TEMP_DATA_DIR, "DeepPCB_repo/PCBData")

    FINAL_DATASET_DIR = "industrial_dataset"
    IMG_DIR = os.path.join(FINAL_DATASET_DIR, "images")
    LBL_DIR = os.path.join(FINAL_DATASET_DIR, "labels")

    # --- Setup Directories ---
    print("Setting up final directory structure...")
    shutil.rmtree(FINAL_DATASET_DIR, ignore_errors=True)
    os.makedirs(os.path.join(IMG_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(IMG_DIR, "val"), exist_ok=True)
    os.makedirs(os.path.join(LBL_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(LBL_DIR, "val"), exist_ok=True)

    temp_neu_extracted = os.path.join(TEMP_DATA_DIR, "neu_extracted")
    os.makedirs(temp_neu_extracted, exist_ok=True)

    all_files_info = []

    # --- Define Class Mappings ---
    neu_classes = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ]
    deeppcb_classes = ["open", "short", "mousebite", "spur", "copper", "pinhole"]
    all_class_names = sorted(
        list(set(neu_classes + deeppcb_classes))
    )  # Use sorted set for consistency
    class_mapping = {name: i for i, name in enumerate(all_class_names)}

    # --- 1. Process NEU-DET ---
    print(f"Processing NEU-DET dataset from {NEU_REPO_PATH}...")

    neu_images_path = os.path.join(NEU_REPO_PATH, "IMAGES")
    neu_annots_path = os.path.join(NEU_REPO_PATH, "ANNOTATIONS")

    print("Converting NEU-DET annotations...")
    for xml_file in tqdm(glob(os.path.join(neu_annots_path, "*.xml"))):
        img_filename = convert_neu_annotation(xml_file, LBL_DIR, class_mapping)
        if not img_filename.lower().endswith(".jpg"):
            img_filename += ".jpg"
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        if os.path.exists(os.path.join(LBL_DIR, label_filename)):
            all_files_info.append(
                {"img_src": os.path.join(neu_images_path, img_filename), "lbl_name": label_filename}
            )

    # --- 2. Process DeepPCB ---
    print(f"Processing DeepPCB dataset from {DEEPPCB_REPO_PATH}...")
    deeppcb_label_files = glob(os.path.join(DEEPPCB_REPO_PATH, "**/*.txt"), recursive=True)

    print("Converting DeepPCB annotations...")
    for txt_file in tqdm(deeppcb_label_files):
        # Filter out non-label files based on DeepPCB repo structure
        if "test.txt" in os.path.basename(txt_file) or "temp.txt" in os.path.basename(txt_file):
            continue

        img_path_test = txt_file.replace(".txt", "_test.jpg")

        if os.path.exists(img_path_test):
            img_filename = convert_deeppcb_annotation(txt_file, LBL_DIR, class_mapping)
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            if os.path.exists(os.path.join(LBL_DIR, label_filename)):
                all_files_info.append({"img_src": img_path_test, "lbl_name": label_filename})

    # --- 3. Split and Move Files ---
    print(f"\nFound {len(all_files_info)} total images with labels. Shuffling and splitting...")
    random.shuffle(all_files_info)

    split_index = int(len(all_files_info) * 0.9)
    train_files = all_files_info[:split_index]
    val_files = all_files_info[split_index:]

    print(f"Moving {len(train_files)} training files...")
    for f_info in tqdm(train_files):
        shutil.copy(f_info["img_src"], os.path.join(IMG_DIR, "train"))
        shutil.move(os.path.join(LBL_DIR, f_info["lbl_name"]), os.path.join(LBL_DIR, "train"))

    print(f"Moving {len(val_files)} validation files...")
    for f_info in tqdm(val_files):
        shutil.copy(f_info["img_src"], os.path.join(IMG_DIR, "val"))
        shutil.move(os.path.join(LBL_DIR, f_info["lbl_name"]), os.path.join(LBL_DIR, "val"))

    # --- 4. Create data.yaml ---
    print("Creating data.yaml file...")
    yaml_content = f"""
# Path to the dataset root directory
path: {os.path.abspath(FINAL_DATASET_DIR)}

# Train/val paths relative to 'path'
train: images/train
val: images/val

# Class names
names:
"""
    for i, name in enumerate(all_class_names):
        yaml_content += f"  {i}: {name}\n"

    with open(os.path.join(FINAL_DATASET_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)

    # --- 5. Cleanup ---
    print("Cleaning up temporary files...")
    shutil.rmtree(LBL_DIR)  # Remove the temporary top-level labels dir

    print("\n--- Dataset preparation complete! ---")
    print(f"New dataset is located at: {os.path.abspath(FINAL_DATASET_DIR)}")
    print(
        f"YAML config file is at: {os.path.abspath(os.path.join(FINAL_DATASET_DIR, 'data.yaml'))}"
    )
    print(f"Total classes: {len(all_class_names)}")
    print(f"Training images: {len(train_files)}, Validation images: {len(val_files)}")


if __name__ == "__main__":
    main()
