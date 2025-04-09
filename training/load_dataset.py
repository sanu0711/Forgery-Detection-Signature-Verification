import os 
import kagglehub
import shutil

# pip install git+https://github.com/Kaggle/kagglehub

def download_dataset():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    path = kagglehub.dataset_download("robinreni/signature-verification-dataset")
    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")
    print("Path to dataset files:", path)
    print("Train path:", train_path)
    print("Test path:", test_path)

    os.makedirs("/content/Train/Fake", exist_ok=True)
    os.makedirs("/content/Train/Real", exist_ok=True)
    os.makedirs("/content/Test/Fake", exist_ok=True)
    os.makedirs("/content/Test/Real", exist_ok=True)

def copy_files_local(src_dir, dst_dir_fake, dst_dir_real):
    print(f"Checking files in {src_dir}")
    for item in os.listdir(src_dir):
        item_path = os.path.join(src_dir, item)
        if os.path.isfile(item_path):  # Check if it's a regular file
            if item.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check if it's a valid image file
                if item.lower().endswith("_forg.jpg") or item.lower().endswith("_forg.jpeg") or item.lower().endswith("_forg.png"):  # Check if filename ends with "_forg"
                    dst_file = os.path.join(dst_dir_fake, item)
                    print(f"Copying {item_path} to {dst_file}")
                    shutil.copy(item_path, dst_file)
                else:
                    dst_file = os.path.join(dst_dir_real, item)
                    print(f"Copying {item_path} to {dst_file}")
                    shutil.copy(item_path, dst_file)
            else:
                print(f"Skipping {item_path}, not a valid image file")
        elif os.path.isdir(item_path):  # Check if it's a directory
            if item.lower().endswith("_forg"):  # Check if directory name ends with "_forg"
                dst_dir = os.path.join(dst_dir_fake, item)  # Destination directory in the "Fake" folder
                print(f"Copying directory {item_path} to {dst_dir}")
                shutil.copytree(item_path, dst_dir, dirs_exist_ok=True)
            else:
                dst_dir = os.path.join(dst_dir_real, item)  # Destination directory in the "Real" folder
                print(f"Copying directory {item_path} to {dst_dir}")
                shutil.copytree(item_path, dst_dir, dirs_exist_ok=True)
        else:
            print(f"Skipping {item_path}, not a file or directory")
            
def train_data_copy():
    train_src_dir = "/content/train"
    train_dst_dir_fake = "/content/Train/Fake"
    train_dst_dir_real = "/content/Train/Real"
    # Copy files for training dataset
    copy_files_local(train_src_dir, train_dst_dir_fake, train_dst_dir_real)
    print("Training dataset copied successfully.")
    return None

def test_data_copy():
    test_src_dir = "/content/test"
    test_dst_dir_fake = "/content/Test/Fake"
    test_dst_dir_real = "/content/Test/Real"
    # Copy files for testing dataset
    copy_files_local(test_src_dir, test_dst_dir_fake, test_dst_dir_real)
    print("Testing dataset copied successfully.")
    return None

def testing_cpy_file():
    train_fake_images = os.listdir("/content/Train/Fake")
    train_real_images = os.listdir("/content/Train/Real")
    test_fake_images = os.listdir("/content/Test/Fake")
    test_real_images = os.listdir("/content/Test/Real")

    # Print the contents of each directory
    print(f"Total Image data: {len(train_fake_images)} Train/Fake Data: {train_fake_images}")
    print(f"Total Image data: {len(train_real_images)} Train/Real Data: {train_real_images}")
    print(f"Total Image data: {len(test_fake_images)} Test/Fake Data: {test_fake_images}")
    print(f"Total Image data: {len(test_real_images)} Test/Real Data: {test_real_images}")

    if len(train_fake_images) == 0 or len(train_real_images) == 0 or len(test_fake_images) == 0 or len(test_real_images) == 0:
        raise ValueError("One of the directories is empty. Please check the paths and ensure images are correctly copied.")

if __name__ == "__main__":
    download_dataset()
    train_data_copy()
    test_data_copy()
    testing_cpy_file()




