# diffusion
Tinkering with Advances in Diffusion / Flow Modeling

**[API Docs](https://tmatteso.github.io/diffusion/)**

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

## Setting up a machine as a GitHub Actions self-hosted runner

The [CUDA devcontainer workflow](.github/workflows/devcontainer-cuda.yml) needs a
GPU-equipped self-hosted runner — GitHub's hosted runners don't have NVIDIA GPUs.
This registers a machine as a runner; it's separate from (and can be done on the
same box as) the devcontainer setup above.

> **Security note:** only register self-hosted runners on private repos, or on
> forks-disabled/trusted-contributor-only public repos. Anyone who can open a PR
> against a public repo with a self-hosted runner can run arbitrary code on that
> machine.

### 1. Generate a registration token

On GitHub: **Settings → Actions → Runners → New self-hosted runner**, select
Linux/x64, and copy the `--token` value shown in the setup snippet (it's
short-lived, so use it right away).

### 2. Create a dedicated non-root user

The runner refuses to be configured as root (`./config.sh` errors with
`Must not run with sudo`) — it wants a low-privilege user so job steps don't
execute as root by default:

```bash
useradd -m -s /bin/bash actions-runner
```

Give it Docker access so it can build/run the CUDA devcontainer without sudo:

```bash
sudo usermod -aG docker actions-runner
```

### 3. Download and configure the runner

Download as the dedicated user (or download as any user, then `chown` the
directory to `actions-runner` before configuring):

```bash
su - actions-runner
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-linux-x64.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.321.0/actions-runner-linux-x64-2.321.0.tar.gz
tar xzf ./actions-runner-linux-x64.tar.gz

./config.sh --url https://github.com/<owner>/<repo> \
  --token <registration-token> \
  --labels cuda-gpu
```

Use the label(s) referenced by `runs-on:` in the workflow — currently
`[self-hosted, cuda-gpu]` — so jobs actually get scheduled onto this
machine. Labels must match exactly: a runner tagged `gpu` will not
satisfy `runs-on: [self-hosted, cuda-gpu]` and vice versa.

> Registration tokens are short-lived and single-use — don't paste them into
> chat logs, issues, or commits. If one leaks, regenerate it from
> **Settings → Actions → Runners** right away.

### 4. Install it as a service

Run this part as root (or with sudo) — it sets up systemd to run the runner
process as the `actions-runner` user, not root, so it survives reboots and SSH
disconnects:

```bash
cd /home/actions-runner/actions-runner
sudo ./svc.sh install actions-runner
sudo ./svc.sh start
sudo ./svc.sh status
```

### 5. Verify it's connected

Check **Settings → Actions → Runners** on GitHub — the machine should show as
**Idle** with the labels you configured. You can also query it via the API:

```bash
gh api repos/<owner>/<repo>/actions/runners \
  --jq '.runners[] | {name, status, labels: [.labels[].name]}'
```

---

## If the NVIDIA Container Toolkit is broken (e.g. after VM maintenance)

Check whether GPUs are visible to Docker at all:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If that fails, re-run the toolkit installation steps in section 1 above.
