import subprocess
import bentoml
import mlflow
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

DOCKER_USERNAME = "jy323"
BENTO_IMAGE_NAME = "loanriskservice"  # Ensure this is your service name in lowercase

# Read the promoted model version from file (set by evaluate.py)
ver_file = "model_version.txt"
if not os.path.isfile(ver_file):
    raise FileNotFoundError("model_version.txt not found. Please run evaluation to promote a model.")
with open(ver_file) as f:
    mlflow_version = f.read().strip()

# Define version tag (e.g., v9) and full image names
version_tag = f"v{mlflow_version}"
FULL_IMAGE_TAG = f"{DOCKER_USERNAME}/{BENTO_IMAGE_NAME}:{version_tag}"
LATEST_TAG     = f"{DOCKER_USERNAME}/{BENTO_IMAGE_NAME}:latest"

def get_newest_image_tag(repo):
    """
    Returns the tag of the most recently built image for the given repository.
    Assumes the containerize command produces one new image.
    """
    output = subprocess.check_output(
        ["docker", "images", repo, "--format", "{{.Tag}}"]
    ).decode().splitlines()
    if output:
        # We assume the first tag is the newest.
        return output[0]
    raise ValueError(f"No images found for repository {repo}")

def main():
    print("Importing MLflow model into Bento store ...")
    bentoml.mlflow.import_model("loan_risk_model", "models:/LoanRiskModel@production")
    
    print("Building Bento ...")
    subprocess.run(["bentoml", "build"], check=True)
    
    print(f"Running 'bentoml containerize {BENTO_IMAGE_NAME}:latest' ...")
    subprocess.run(["bentoml", "containerize", f"{BENTO_IMAGE_NAME}:latest"], check=True)
    
    # Retrieve the actual tag created by BentoML (a random hash, not necessarily "latest")
    newest_tag = get_newest_image_tag(BENTO_IMAGE_NAME)
    print(f"Found built image tag: {newest_tag}")
    
    # Tag the built image with the versioned tag and also as latest
    subprocess.run(
        ["docker", "tag", f"{BENTO_IMAGE_NAME}:{newest_tag}", FULL_IMAGE_TAG], check=True
    )
    subprocess.run(
        ["docker", "tag", f"{BENTO_IMAGE_NAME}:{newest_tag}", LATEST_TAG], check=True
    )
    
    print(f"Pushing image {FULL_IMAGE_TAG} to Docker Hub ...")
    subprocess.run(["docker", "push", FULL_IMAGE_TAG], check=True)
    print(f"Pushing image {LATEST_TAG} to Docker Hub ...")
    subprocess.run(["docker", "push", LATEST_TAG], check=True)
    
    print(f"✅ Docker images pushed as {FULL_IMAGE_TAG} and {LATEST_TAG}")

if __name__ == "__main__":
    main()
