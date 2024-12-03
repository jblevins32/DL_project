import os
import urllib.request
import zipfile

class KittiDataDownloader:

    dataset_dir = "./dataset"
    image_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")

    dataset_images_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip"
    dataset_training_labels_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip"

    def dataExists(self):
        return os.path.exists(self.dataset_dir)

    def prepareDataset(self):
        if not self.dataExists():
            # make dataset folder
            os.makedirs(self.dataset_dir)

            # populate folder with downloaded dataset images
            self.downloadImages()

            # populate folder with downloaded training ground truth labels
            self.downloadTrainingLabels()
        else:
            print(f"\nDirectory {self.dataset_dir} already exists. No download needed.\n "
                  f"** If dataset has been corrupted, delete the dataset folder to trigger re-download.**\n")

    def downloadImages(self):
        os.makedirs(self.image_dir)
        self.downloadContentsOfURL(self.dataset_images_url, self.image_dir)

    def downloadTrainingLabels(self):
        os.makedirs(self.label_dir)
        self.downloadContentsOfURL(self.dataset_training_labels_url, self.label_dir)

    def downloadContentsOfURL(self, url, directoryPath):
        datasetName = url.split("/")[-1]
        dataset_path = os.path.join(directoryPath, datasetName)

        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, dataset_path)
        print(f"Dataset downloaded and saved to {dataset_path}")

        # Unzip the dataset
        print("Unzipping the dataset...")
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(directoryPath)
        print(f"Dataset extracted to {directoryPath}")

        # Optionally, remove the zip file after extraction
        os.remove(dataset_path)
        print(f"Removed the zip file: {dataset_path}")