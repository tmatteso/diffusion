# Setting up a machine as a GitHub Actions self-hosted runner

The [CUDA devcontainer workflow](../.github/workflows/devcontainer-cuda.yml) needs a
GPU-equipped self-hosted runner — GitHub's hosted runners don't have NVIDIA GPUs.
This registers a machine as a runner; it's separate from (and can be done on the
same box as) the devcontainer setup in [first_time_instructions.md](first_time_instructions.md).

> **Security note:** only register self-hosted runners on private repos, or on
> forks-disabled/trusted-contributor-only public repos. Anyone who can open a PR
> against a public repo with a self-hosted runner can run arbitrary code on that
> machine.

## 1. Generate a registration token

On GitHub: **Settings → Actions → Runners → New self-hosted runner**, select
Linux/x64, and copy the `--token` value shown in the setup snippet (it's
short-lived, so use it right away).

## 2. Create a dedicated non-root user

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

## 3. Download and configure the runner

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

## 4. Install it as a service

Run this part as root (or with sudo) — it sets up systemd to run the runner
process as the `actions-runner` user, not root, so it survives reboots and SSH
disconnects:

```bash
cd /home/actions-runner/actions-runner
sudo ./svc.sh install actions-runner
sudo ./svc.sh start
sudo ./svc.sh status
```

## 5. Verify it's connected

Check **Settings → Actions → Runners** on GitHub — the machine should show as
**Idle** with the labels you configured. You can also query it via the API:

```bash
gh api repos/<owner>/<repo>/actions/runners \
  --jq '.runners[] | {name, status, labels: [.labels[].name]}'
```
