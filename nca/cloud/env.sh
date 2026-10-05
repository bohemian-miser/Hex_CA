# shellcheck shell=bash
# Settings for every script in nca/cloud (sourced, never run). Each can be overridden from the environment,
# e.g. BUCKET=/tmp/fake-bucket nca/cloud/pull.sh to test against a local directory.
# PROJECT is shared with another product: every gcloud command passes --project, every resource we create
# carries the label purpose=hexca, and the "is anything left" checks filter on that label.

PROJECT=${PROJECT:-recipe-lanes-staging}
REGION=${REGION:-us-central1}
ZONE=${ZONE:-us-central1-a}                          # tried first
ZONE_FALLBACKS=${ZONE_FALLBACKS:-us-central1-b us-central1-c}  # then these, if ZONE has no Spot L4 capacity
BUCKET=${BUCKET:-gs://recipe-lanes-staging-hexca-runs}  # private; the VM's service account may only use this
SA=${SA:-hexca-vm@recipe-lanes-staging.iam.gserviceaccount.com}
MACHINE=${MACHINE:-g2-standard-4}                    # 1x L4 24 GB, 4 vCPU, 16 GB RAM
VM=${VM:-hexca-train}
LABEL=${LABEL:-purpose=hexca}
MAX_HOURS=${MAX_HOURS:-12}          # --max-run-duration: the VM is deleted after this long, whatever happens
BUDGET_HOURS=${BUDGET_HOURS:-45}  # launch.sh refuses if the ledger's VM hours + MAX_HOURS would pass this
SETUP_MIN=${SETUP_MIN:-25}         # boot + driver install (and its reboot) + venv + clone, before training
REPO=${REPO:-https://github.com/bohemian-miser/Hex_CA.git}  # the VM clones this, at the launched commit
TORCH=${TORCH:-torch}  # pip args for the VM's torch, e.g. "torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126" (no commas: it travels in --metadata)
# Boot image. The Deep Learning VM "common" image has the NVIDIA driver preinstalled, which saves the driver
# install and its reboot (startup.sh installs the driver only when nvidia-smi is missing). Plain Ubuntu also works:
#   IMAGE_FAMILY=ubuntu-2404-lts-amd64 IMAGE_PROJECT=ubuntu-os-cloud
IMAGE_FAMILY=${IMAGE_FAMILY:-common-cu129-ubuntu-2404-nvidia-580}
IMAGE_PROJECT=${IMAGE_PROJECT:-deeplearning-platform-release}
# Budget (owner, 2026-10-05 ~12:00 UTC): $20 of compute until 2026-10-06 ~12:00 UTC, everything since the first
# launch included. g2-standard-4 Spot is ~$0.41/h with disk and address, so 45 VM-hours is ~$18.5; one GPU at a
# time (quota 1) means the 24-hour window, not the dollars, is the real limit. A bigger machine type costs more
# per hour: lower BUDGET_HOURS to match (g2-standard-8 ~$0.49/h -> 40 h).
