import os
import subprocess

DOCKER_USER = "jy323"
SERVICE_NAME = "loanriskservice"

def main():
    # Read version
    version_file = "model_version.txt"
    if not os.path.exists(version_file):
        raise FileNotFoundError("model_version.txt not found.")

    with open(version_file, "r") as f:
        version = f.read().strip()

    full_image_tag = f"{DOCKER_USER}/{SERVICE_NAME}:v{version}"
    print(f"Using Docker image: {full_image_tag}")

    # Read Seldon template
    with open("k8s/seldon_deployment.yaml", "r") as f:
        template = f.read()

    # Replace placeholder(s)
    final_yaml = template.replace("${IMAGE_TAG}", full_image_tag)

    # Write to temp file
    temp_path = "k8s/temp_deployment.yaml"
    with open(temp_path, "w") as f:
        f.write(final_yaml)

    # Apply
    print("🚀 Deploying to Seldon...")
    subprocess.run(["kubectl", "apply", "-f", temp_path], check=True)
    print("✅ Deployment applied successfully.")

if __name__ == "__main__":
    main()
