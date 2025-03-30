import os
import subprocess

DOCKER_USER = "jy323"
SERVICE_NAME = "loanriskservice"
DEPLOY_NAMESPACE = "ml"

def main():
    version_file = "model_version.txt"
    if not os.path.exists(version_file):
        raise FileNotFoundError("model_version.txt not found.")

    with open(version_file, "r") as f:
        version = f.read().strip()

    image_tag = f"{DOCKER_USER}/{SERVICE_NAME}:v{version}"
    print(f"Using Docker image: {image_tag}")

    # Read and render the k8s deployment template
    with open("k8s/bento_deployment.yaml", "r") as f:
        template = f.read()

    rendered_yaml = template.replace("${IMAGE_TAG}", image_tag)

    # Save to a temp file
    temp_yaml = "k8s/rendered_deployment.yaml"
    with open(temp_yaml, "w") as f:
        f.write(rendered_yaml)

    # Create namespace if needed
    subprocess.run(["kubectl", "create", "namespace", DEPLOY_NAMESPACE], stderr=subprocess.DEVNULL)

    # Apply the deployment
    print("🚀 Deploying BentoML service to Kubernetes...")
    subprocess.run(["kubectl", "apply", "-f", temp_yaml, "-n", DEPLOY_NAMESPACE], check=True)
    print("✅ Deployment complete.")

if __name__ == "__main__":
    main()
