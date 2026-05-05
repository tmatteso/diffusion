# diffusion
Tinkering with Advances in Diffusion / Flow Modeling

---

## Setting up the devcontainer on a new CUDA machine

### 1. Install the NVIDIA Container Toolkit

The toolkit is required for Docker to pass GPUs into containers via `--gpus=all`. Run these on the host machine:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify the toolkit is working:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### 2. Configure git

The devcontainer bind-mounts `~/.gitconfig` from the host, so git must be configured before opening the container:

```bash
touch ~/.gitconfig
git config --global user.name "tmatteso"
git config --global user.email "tlmattesonr@gmail.com"
```

### 3. Open the devcontainer

In VS Code, open the command palette and select **Dev Containers: Reopen in Container**, then choose the **CUDA** configuration (`.devcontainer/cuda/devcontainer.json`).

### 4. Log in to Weights & Biases

Inside the container:

```bash
wandb login
# paste your API key when prompted
```

### 5. Copy training data onto the machine

From your local machine, transfer data with `scp`:

```bash
scp -r -P <port> -i ~/.ssh/id_ed25519_vast_ai \
  /Users/tomasmatteson/diffusion/pallatom/data \
  root@<host-ip>:/root/diffusion/pallatom/
```

---

## If the NVIDIA Container Toolkit is broken (e.g. after VM maintenance)

Check whether GPUs are visible to Docker at all:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If that fails, re-run the toolkit installation steps in section 1 above.
