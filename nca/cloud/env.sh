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
MAX_HOURS=${MAX_HOURS:-6}          # --max-run-duration: the VM is deleted after this long, whatever happens
BUDGET_HOURS=${BUDGET_HOURS:-6.5}  # launch.sh refuses if the ledger's VM hours + MAX_HOURS would pass this
SETUP_MIN=${SETUP_MIN:-25}         # boot + driver install (and its reboot) + venv + clone, before training
REPO=${REPO:-https://github.com/bohemian-miser/Hex_CA.git}  # the VM clones this, at the launched commit
TORCH=${TORCH:-torch}  # pip args for the VM's torch, e.g. "torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126" (no commas: it travels in --metadata)
