import os

from huggingface_hub import hf_hub_download, list_repo_files


def download_boom_benchmark(local_path: str) -> str:
    """
    Download all raw files of the Boom Benchmark dataset from Hugging Face into a local directory.

    Args:
        local_path (str): Parent directory where 'boom_benchmark' folder will be created. Must exist.

    Returns:
        str: Path to the downloaded dataset folder.
    """
    repo_id = "Datadog/BOOM"
    dataset_dir = os.path.join(local_path, "boom_benchmark")

    if not os.path.isdir(local_path):
        raise ValueError(f"Local path must exist and be a directory: {local_path}")

    if os.path.exists(dataset_dir):
        print(f"Boom Benchmark already exists at {dataset_dir}.")
    else:
        print("Downloading all raw files from Hugging Face...")
        os.makedirs(dataset_dir, exist_ok=True)

        # List all files in the dataset repository
        files = list_repo_files(repo_id, repo_type="dataset")

        for file in files:
            local_file_path = hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=dataset_dir,
                local_dir_use_symlinks=False,
                repo_type="dataset",
            )
            print(f"  - Downloaded: {file} → {local_file_path}")

    # Set environment variable
    os.environ["BOOM"] = dataset_dir
    return dataset_dir
