import os
import torch
from torch.utils.data import Dataset
from PIL import Image

CLASS_MAPPING = {
    "Car": 0,
    "Van": 1,
    "Pedestrian": 2,
    "Cyclist": 3,
}

class KittiDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the root directory of training images.
            label_dir (str): Path to the root directory of label text files.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.samples = []

        self.desiredWidth = 1220
        self.desiredHeight = 365

        self.shouldCrop = False

        # Parse all scenes and frames
        scenes = sorted(os.listdir(image_dir))
        for scene in scenes:
            scene_image_dir = os.path.join(image_dir, scene)
            scene_label_dir = os.path.join(label_dir, f"{scene}.txt")
            cropped_image_dir = os.path.join(scene_image_dir, 'cropped')

            if not os.path.exists(cropped_image_dir):
                os.makedirs(cropped_image_dir)
            else:
                self.shouldCrop = False

            if not os.path.exists(scene_label_dir):
                continue

            # Parse the label file for this scene
            with open(scene_label_dir, 'r') as f:
                labels = [line.strip().split() for line in f.readlines()]

            # Create a mapping of frame indices to labels
            frame_labels = {}
            for label in labels:
                frame_idx = int(label[0])
                bbox = list(map(float, label[6:10]))
                class_name = label[2]
                class_id = CLASS_MAPPING.get(class_name, -1)  # Default to -1 for unknown classes
                if class_id != -1:
                    if frame_idx not in frame_labels:
                        frame_labels[frame_idx] = []
                    frame_labels[frame_idx].append([bbox, class_id])

            # Associate frames with labels
            frame_files = sorted(os.listdir(scene_image_dir))
            for frame_file in frame_files:
                if frame_file == "cropped":
                    continue
                
                frame_idx = int(os.path.splitext(frame_file)[0])
                if frame_idx in frame_labels:
                    cropped_image_path = os.path.join(cropped_image_dir, frame_file)

                    if self.shouldCrop:
                        image_path = os.path.join(scene_image_dir, frame_file)
                        self.cropImage(image_path, cropped_image_path)

                    # Update the samples with the new image path
                    self.samples.append({
                        "image_path": cropped_image_path,
                        "labels": frame_labels[frame_idx],
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Format the labels
        bboxes = torch.tensor([label[["bbox"]] for label in sample["labels"]], dtype=torch.float32)
        class_ids = torch.tensor([label["class_id"] for label in sample["labels"]], dtype=torch.long)

        return image, {"bboxes": bboxes, "class_ids": class_ids}


    def cropImage(self, imagePath, croppedImagePath):
        # Load the image
        image = Image.open(imagePath)
        width, height = image.size

        # Calculate coordinates for cropping the center
        left = (width - self.desiredWidth) / 2
        upper = (height - self.desiredHeight) / 2
        right = (width + self.desiredWidth) / 2
        lower = (height + self.desiredHeight) / 2

        # Adjust coordinates if they are out of bounds
        left = max(0, left)
        upper = max(0, upper)
        right = min(width, right)
        lower = min(height, lower)

        # Crop and resize the image
        image_cropped = image.crop((left, upper, right, lower))
        image_cropped = image_cropped.resize((self.desiredWidth, self.desiredHeight))

        # Save the cropped image
        image_cropped.save(croppedImagePath)