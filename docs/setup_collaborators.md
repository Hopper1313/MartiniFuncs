# MLIP Pair Builder: Setup Guide for Collaborators

Needs Linux, or Windows with WSL2. Run everything in the Linux terminal.

## 1. WSL2 (Windows only)

In an admin PowerShell, then reboot and open Ubuntu:

```bash
wsl --install -d Ubuntu
```

## 2. Apptainer

```bash
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update && sudo apt install -y apptainer git curl
```

## 3. Repo and image

```bash
git clone https://github.com/Hopper1313/MartiniFuncs.git ~/MartiniFuncs
mkdir -p ~/containers && cd ~/containers
R=https://github.com/Hopper1313/MartiniFuncs/releases/download/mlip-b742b5b
curl -LO $R/mlip.sif && curl -LO $R/mlip.sif.sha256 && sha256sum -c mlip.sif.sha256
```

## 4. Models

```bash
cd ~/MartiniFuncs
export HF_TOKEN=hf_...   # UMA only: request access at huggingface.co/facebook/UMA first
bash containers/mlip/fetch_models.sh --accept-asl
```

No UMA: `bash containers/mlip/fetch_models.sh --accept-asl mace aimnet`. `--accept-asl` confirms MACE-OMOL's [Academic Software License](https://github.com/gabor1/ASL) covers your use.

## 5. Run

```bash
cd ~/MartiniFuncs && bash containers/mlip/run_server.sh   # --ncpus N to limit CPUs (default: all)
```

Open http://127.0.0.1:8000/. Details: [containers/mlip/README.md](../containers/mlip/README.md).
