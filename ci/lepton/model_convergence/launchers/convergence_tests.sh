#!/usr/bin/env bash
set -euo pipefail

# Get job name
JOB_NAME="${LEPTON_JOB_NAME:-unknown-job}"

# Log the hydra config at the start
echo "=========================================="
echo "HYDRA CONFIG FOR JOB: $JOB_NAME"
echo "=========================================="
echo '__ALL_CONFIG_JSON__' | jq '.' 2>/dev/null || echo '__ALL_CONFIG_JSON__'
echo "=========================================="
echo ""

# Run the training script
set +e
(
__SCRIPT__
)
RC=$?
set -e

echo "commit in bionemo-framework"
(cd bionemo-framework && git log -1 || true)
# Always grab the exact commit currently checked out in the framework repo
COMMIT_SHA="$(cd bionemo-framework && git rev-parse HEAD 2>/dev/null || true)"
echo "Resolved framework commit: ${COMMIT_SHA:-<none>}"

# Authenticate to Lepton
pip install -q leptonai >/dev/null 2>&1 || pip install -q leptonai || true
lep login -c "$LEP_LOGIN_CREDENTIALS" || true

# Get lepton job details (extract JSON from CLI output)
JOB_INFO="$(
  lep job get --id "$JOB_NAME" 2>/dev/null \
  | awk '
    BEGIN { json=""; depth=0; started=0 }
    {
      for (i=1; i<=length($0); i++) {
        ch = substr($0, i, 1)
        if (ch == "{") { depth++; started=1 }
        if (started)      json = json ch
        if (ch == "}") {
          depth--
          if (started && depth == 0) { print json; exit }
        }
      }
    }' \
  | jq -c '
    {
      metadata: {
        id: .metadata.id,
        name: .metadata.name,
        created_at: .metadata.created_at,
        created_by: .metadata.created_by,
        owner: .metadata.owner,
        visibility: .metadata.visibility
      },
      spec: {
        resource_shape: .spec.resource_shape,
        affinity: {
          allowed_dedicated_node_groups: .spec.affinity.allowed_dedicated_node_groups,
          allowed_nodes_in_node_group: .spec.affinity.allowed_nodes_in_node_group
        },
        container_image: .spec.container.image,
        completions: .spec.completions,
        parallelism: .spec.parallelism,
        envs: .spec.envs,
        mounts: .spec.mounts,
        image_registry_auth: .spec.image_pull_secrets,
        ttl_seconds_after_finished: .spec.ttl_seconds_after_finished,
        log_enable_collection: .spec.log.enable_collection
      },
      status: {
        job_name: .status.job_name,
        state: .status.state,
        ready: .status.ready,
        active: .status.active,
        failed: .status.failed,
        succeeded: .status.succeeded,
        creation_time: .status.creation_time,
        completion_time: .status.completion_time
      }
    }
  ' 2>/dev/null
)"
JOB_INFO_JSON="$(printf '%s' "$JOB_INFO" | jq -c . 2>/dev/null || echo '{}')"

# Ingest provided config JSON
ALL_CONFIG_JSON='__ALL_CONFIG_JSON__'
if echo "$ALL_CONFIG_JSON" | jq -e . >/dev/null 2>&1; then
  ALL_CONFIG_JSON_UPDATED="$(printf '%s' "$ALL_CONFIG_JSON" | jq -c '.')"
else
  echo "Warning: ALL_CONFIG_JSON is not valid JSON. Using empty object."
  ALL_CONFIG_JSON_UPDATED='{}'
fi

# Inject/overwrite the resolved framework commit (only if we actually got one)
if [ -n "${COMMIT_SHA:-}" ]; then
  ALL_CONFIG_JSON_UPDATED="$(printf '%s' "$ALL_CONFIG_JSON_UPDATED" | jq -c --arg commit "$COMMIT_SHA" '.commit_sha = $commit')"

  # Find which branch contains this commit
  RESOLVED_BRANCH="$(cd bionemo-framework && git branch -r --contains "$COMMIT_SHA" | grep 'origin/' | head -1 | sed 's|.*origin/||' || true)"

  if [ -n "$RESOLVED_BRANCH" ] && [ "$RESOLVED_BRANCH" != "HEAD" ]; then
    ALL_CONFIG_JSON_UPDATED="$(printf '%s' "$ALL_CONFIG_JSON_UPDATED" | jq -c --arg branch "$RESOLVED_BRANCH" '.branch = $branch')"
  fi
fi

# Extract values from config (with sensible defaults)
RECIPE_SUBDIR="$(printf '%s' "$ALL_CONFIG_JSON_UPDATED" | jq -r '.recipe_subdir // "esm2_native_te"')"
KRATOS_SUBJECT="$(printf '%s' "$ALL_CONFIG_JSON_UPDATED" | jq -r '.kratos_subject // "convergence_tests_v0.0.1"')"



# ---------------------------
# Collect NVIDIA SMI as JSON (no cuda_version in --query-gpu)
# ---------------------------
set +e
NVIDIA_SMI_BIN="$(command -v nvidia-smi || echo /usr/bin/nvidia-smi)"
NVIDIA_SMI_JSON="[]"
for GPU_FIELDS in \
'index,uuid,name,driver_version,pci.bus_id,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,clocks.gr,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory,compute_mode' \
'index,uuid,name,driver_version,pci.bus_id,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,clocks.current.graphics,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory,compute_mode' \
'index,uuid,name,driver_version,pci.bus_id,memory.total,memory.free,memory.used,utilization.gpu'; do
  RAW_SMI="$("$NVIDIA_SMI_BIN" --query-gpu="$GPU_FIELDS" --format=csv,noheader,nounits 2>/dev/null || true)"
  if [ -n "$RAW_SMI" ]; then
    NVIDIA_SMI_JSON="$(
      GPU_FIELDS="$GPU_FIELDS" python3 - <<'PY' 2>/dev/null || true
import os, sys, csv, json
keys = [s.strip() for s in os.environ.get("GPU_FIELDS","").split(",") if s.strip()]
rows = []
for r in csv.reader(sys.stdin):
    if not r:
        continue
    vals = [x.strip() for x in r]
    if len(vals) < len(keys):
        vals += [None]*(len(keys)-len(vals))
    rows.append(dict(zip(keys, vals[:len(keys)])))
print(json.dumps(rows))
PY
    <<< "$RAW_SMI"
    )"
    if [ -n "$NVIDIA_SMI_JSON" ] && [ "$NVIDIA_SMI_JSON" != "[]" ]; then
      break
    fi
  fi
done

RAW_APPS="$("$NVIDIA_SMI_BIN" --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
if [ -n "$RAW_APPS" ]; then
  NVIDIA_COMPUTE_APPS_JSON="$(
    python3 - <<'PY' 2>/dev/null || true
import sys, csv, json
rows=[]
for r in csv.reader(sys.stdin):
    if not r:
        continue
    gpu_uuid = r[0].strip() if len(r)>0 else None
    # pid as int where possible
    pid = None
    if len(r)>1:
        try: pid = int(r[1].strip())
        except: pid = None
    process  = r[2].strip() if len(r)>2 else None
    used_mem = r[3].strip() if len(r)>3 else None
    rows.append({"gpu_uuid": gpu_uuid, "pid": pid, "process_name": process, "used_memory": used_mem})
print(json.dumps(rows))
PY
    <<< "$RAW_APPS"
  )"
else
  NVIDIA_COMPUTE_APPS_JSON="[]"
fi

# Driver/CUDA at top level from -q (stable across versions)
DRIVER_VERSION="$("$NVIDIA_SMI_BIN" -q 2>/dev/null | awk -F': ' '/Driver Version/ {print $2; exit}')"
CUDA_VERSION="$("$NVIDIA_SMI_BIN" -q 2>/dev/null | awk -F': ' '/CUDA Version/ {print $2; exit}')"
NVIDIA_DRIVER_INFO="$(jq -n --arg dv "$DRIVER_VERSION" --arg cv "$CUDA_VERSION" 'def nn($x): if ($x|length)>0 then $x else null end; {driver_version: nn($dv), cuda_version: nn($cv)}' 2>/dev/null || echo '{}')"
set -e

# Look for W&B files
WANDB_DIR="/workspace/bionemo-framework/recipes/$RECIPE_SUBDIR/wandb"

WANDB_FOUND=0
WANDB_SUMMARY=""
WANDB_METADATA=""

if [ -d "$WANDB_DIR" ]; then
    if [ -L "$WANDB_DIR/latest-run" ]; then
        LATEST_RUN="$WANDB_DIR/latest-run"
    else
        LATEST_RUN=$(ls -td "$WANDB_DIR"/run-* "$WANDB_DIR"/offline-run-* 2>/dev/null | head -n1)
    fi
    WANDB_ID="$(basename "$(readlink -f "$LATEST_RUN")" | cut -d'-' -f3)"

    echo "WANDB_ID: $WANDB_ID"
    if [ -n "$LATEST_RUN" ] && [ -d "$LATEST_RUN/files" ]; then
        if [ -f "$LATEST_RUN/files/wandb-summary.json" ]; then
            WANDB_SUMMARY="$LATEST_RUN/files/wandb-summary.json"
            WANDB_METADATA="$LATEST_RUN/files/wandb-metadata.json"
            WANDB_FOUND=1
        fi
    fi
fi

if [ "$WANDB_FOUND" = "1" ] && [ -n "$WANDB_SUMMARY" ]; then
    echo "Uploading W&B metrics to Kratos..."

    METADATA_JSON=$(cat "$WANDB_METADATA" 2>/dev/null || echo '{}')
    SUMMARY_JSON=$(cat "$WANDB_SUMMARY" 2>/dev/null || echo '{}')

    COMBINED_JSON=$(jq -n \
        --arg m "$METADATA_JSON" \
        --arg s "$SUMMARY_JSON" \
        --arg wandb_id "$WANDB_ID" \
        --argjson job_info "$JOB_INFO_JSON" \
        --argjson all_config "$ALL_CONFIG_JSON_UPDATED" \
        --argjson nvidia_smi "$NVIDIA_SMI_JSON" \
        --argjson nvidia_compute_apps "$NVIDIA_COMPUTE_APPS_JSON" \
        --argjson nvidia_driver "$NVIDIA_DRIVER_INFO" \
        '
        . + {
          job_name: env.LEPTON_JOB_NAME,
          wandb_id: $wandb_id,
          metadata: ($m | fromjson? // {}),
          summary:  ($s | fromjson? // {}),
          job_info: $job_info,
          config: $all_config,
          nvidia_smi: $nvidia_smi,
          nvidia_compute_apps: $nvidia_compute_apps,
          nvidia_driver: $nvidia_driver
        }
        ')

    echo "$COMBINED_JSON" > wandb-combined.json

    UUID=$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid 2>/dev/null || echo "$(date +%s)-$-$RANDOM")
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")

    if [ -z "${KRATOS_SSA_CLIENT_ID:-}" ] || [ -z "${KRATOS_SSA_SECRET:-}" ] || [ -z "${KRATOS_SSA_URL:-}" ]; then
        echo "Warning: Kratos credentials not found. Skipping telemetry upload."
    else
        ENCODED_CREDS=$(echo -n "${KRATOS_SSA_CLIENT_ID}:${KRATOS_SSA_SECRET}" | base64 | tr -d '\n')
        TOKEN_RESPONSE=$(curl -sS --request POST \
            -H "Content-Type: application/x-www-form-urlencoded" \
            -H "Authorization: Basic $ENCODED_CREDS" \
            "https://${KRATOS_SSA_URL}/token?grant_type=client_credentials&scope=telemetry-write" 2>&1)
        ACCESS_TOKEN=$(echo "$TOKEN_RESPONSE" | jq -r '.access_token' 2>/dev/null)

        if [ -n "$ACCESS_TOKEN" ] && [ "$ACCESS_TOKEN" != "null" ]; then
            JSON_PAYLOAD=$(jq -n \
                --arg id "$UUID" \
                --arg time "$TIMESTAMP" \
                --arg source "bionemo-wandb-logs" \
                --arg type "wandb-training-metrics" \
                --arg subject "$KRATOS_SUBJECT" \
                --argjson data "$COMBINED_JSON" \
                '{
                  "specversion": "1.0",
                  "id": $id,
                  "time": $time,
                  "source": $source,
                  "type": $type,
                  "subject": $subject,
                  "data": $data
                }')

            RESPONSE=$(curl -sS --request POST \
                -H "Content-Type: application/cloudevents+json" \
                -H "Authorization: Bearer ${ACCESS_TOKEN}" \
                "https://prod.analytics.nvidiagrid.net/api/v2/topic/bionemo-convergence-lepton-logs-kratos.telemetry.lepton-poc-v001.prod" \
                --data "$JSON_PAYLOAD" 2>&1)

            if [ $? -eq 0 ]; then
                echo "✓ Event sent successfully to Kratos (ID: $UUID)"
            else
                echo "Failed to send event to Kratos: $RESPONSE"
            fi
        else
            echo "Error: Failed to get Kratos access token"
        fi
    fi
else
    echo "W&B metrics not found - skipping Kratos upload"
fi

exit "$RC"
