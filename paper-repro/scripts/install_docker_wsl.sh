#!/bin/bash
# Install Docker Engine + NVIDIA Container Toolkit inside WSL2 Ubuntu, so containers can use the GPU.
# Safe to rerun, and safe on top of a Docker install made with Docker's own instructions.
# Run with:  sudo bash scripts/install_docker_wsl.sh
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
TARGET_USER=${SUDO_USER:-$USER}

# Remove a duplicate Docker source left by an earlier version of this script. This must happen
# before any apt command: apt refuses to run when two entries for the same repository name
# different keys.
if [ -f /etc/apt/sources.list.d/docker.sources ] && grep -qs "keyrings/docker.gpg" /etc/apt/sources.list.d/docker.list; then
  rm -f /etc/apt/sources.list.d/docker.list /etc/apt/keyrings/docker.gpg
  echo "removed duplicate Docker source (docker.list)"
fi

apt-get update
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings

# Only add Docker's repository if no source for it exists yet (Docker's own install guide adds one).
if ! grep -rqs "download.docker.com" /etc/apt/sources.list /etc/apt/sources.list.d/; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release; echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list
fi

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /etc/apt/keyrings/nvidia-container-toolkit.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin nvidia-container-toolkit

nvidia-ctk runtime configure --runtime=docker
usermod -aG docker "$TARGET_USER"
service docker restart || systemctl restart docker

echo
echo "Done. Check the GPU from a container with:"
echo "  sg docker -c 'docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi'"
