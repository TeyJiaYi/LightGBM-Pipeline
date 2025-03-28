import os
import subprocess

def main():
    # Read the model version from file
    version_file = "model_version.txt"
    if not os.path.exists(version_file):
        raise FileNotFoundError("model_version.txt not found.")
    
    with open(version_file, "r") as f:
        version = f.read().strip()
    
    # Prepend 'v' to form the full version tag (e.g., "v9")
    model_version = f"v{version}"
    print(f"Using model version: {model_version}")

    # Read the Kubernetes deployment template
    deployment_template = "k8s/deployment.yaml"
    with open(deployment_template, "r") as f:
        content = f.read()

    # Replace the placeholder with the actual model version
    content = content.replace("${MODEL_VERSION}", model_version)

    # Write out a temporary deployment file
    temp_deployment = "k8s/temp_deployment.yaml"
    with open(temp_deployment, "w") as f:
        f.write(content)
    
    # Apply the deployment via kubectl
    print("Deploying to Kubernetes...")
    subprocess.run(["kubectl", "apply", "-f", temp_deployment], check=True)
    print("Deployment applied successfully.")

if __name__ == "__main__":
    main()
