#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${repo_root}/model/source/resnet18"
model_file="${source_dir}/resnet18_int8_channelwise.tflite"
readme_file="${source_dir}/README.md"
metadata_file="${source_dir}/model_metadata.json"

model_url="${MODEL_URL:-https://huggingface.co/litert-community/resnet18/resolve/main/resnet18_int8_channelwise.tflite}"
readme_url="${README_URL:-https://huggingface.co/litert-community/resnet18/raw/main/README.md}"
model_revision="${MODEL_REVISION:-main}"
spike_bin="${SPIKE_BIN:-/home/andy/src/riscv-isa-sim/build/spike}"
spike_source="${SPIKE_SOURCE:-/home/andy/src/riscv-isa-sim}"

expected_model_sha256="${MODEL_SHA256:-8fea1acb62829e120b4953b6fbf0f842e0032909254f51c2d989703b66534683}"
expected_readme_sha256="${README_SHA256:-b7f5d7d636ff1f50b6b1b4bf870e143e67d563ab3352666a867a04d950180d68}"

die() {
    printf 'freeze_resnet18.sh: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 ||
        die "required command not found: $1"
}

require_command curl
require_command sha256sum
require_command python3
require_command git

[[ -x "${spike_bin}" ]] ||
    die "Spike executable is not executable: ${spike_bin}"
[[ -d "${spike_source}/.git" ]] ||
    die "Spike source repository not found: ${spike_source}"

mkdir -p "${source_dir}"

download_and_verify() {
    local url="$1"
    local destination="$2"
    local expected="$3"
    local temporary="${destination}.download"
    local actual

    curl --fail --location --retry 3 --silent --show-error \
        --output "${temporary}" "${url}"
    actual="$(sha256sum "${temporary}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] ||
        die "SHA-256 mismatch for ${destination}: expected ${expected}, got ${actual}"
    mv "${temporary}" "${destination}"
}

download_and_verify "${model_url}" "${model_file}" "${expected_model_sha256}"
download_and_verify "${readme_url}" "${readme_file}" "${expected_readme_sha256}"

python_version="$(python3 --version 2>&1 | awk '{print $2}')"
gcc_version="$(
    riscv64-unknown-elf-gcc --version 2>/dev/null |
        awk 'NR == 1 { print $NF }'
)"
[[ -n "${gcc_version}" ]] || die "riscv64-unknown-elf-gcc is not available"

spike_help="$("${spike_bin}" -h 2>&1 | head -n 1)"
spike_version="$(printf '%s\n' "${spike_help}" | sed -n 's/^Spike RISC-V ISA Simulator //p')"
[[ -n "${spike_version}" ]] ||
    die "could not identify Spike from: ${spike_help}"

spike_revision="$(git -C "${spike_source}" rev-parse HEAD)"
spike_branch="$(git -C "${spike_source}" branch --show-current)"
[[ -n "${spike_branch}" ]] ||
    die "Spike checkout is not on a named branch"

model_sha256="$(sha256sum "${model_file}" | awk '{print $1}')"
readme_sha256="$(sha256sum "${readme_file}" | awk '{print $1}')"
downloaded_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

MODEL_REVISION="${model_revision}" \
MODEL_URL="${model_url}" \
README_URL="${readme_url}" \
DOWNLOADED_AT_UTC="${downloaded_at_utc}" \
MODEL_SHA256="${model_sha256}" \
README_SHA256="${readme_sha256}" \
PYTHON_VERSION="${python_version}" \
GCC_VERSION="${gcc_version}" \
SPIKE_VERSION="${spike_version}" \
SPIKE_REVISION="${spike_revision}" \
SPIKE_BRANCH="${spike_branch}" \
SPIKE_BIN="${spike_bin}" \
python3 - "${metadata_file}" <<'PY'
import importlib.metadata
import json
import os
import sys

metadata_file = sys.argv[1]

def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"

metadata = {
    "repository": "litert-community/resnet18",
    "revision": os.environ["MODEL_REVISION"],
    "model_file": "resnet18_int8_channelwise.tflite",
    "model_url": os.environ["MODEL_URL"],
    "readme_file": "README.md",
    "readme_url": os.environ["README_URL"],
    "downloaded_at_utc": os.environ["DOWNLOADED_AT_UTC"],
    "sha256": {
        "resnet18_int8_channelwise.tflite": os.environ["MODEL_SHA256"],
        "README.md": os.environ["README_SHA256"],
    },
    "tools": {
        "python": os.environ["PYTHON_VERSION"],
        "numpy": package_version("numpy"),
        "Pillow": package_version("Pillow"),
        "huggingface_hub": package_version("huggingface_hub"),
        "ai-edge-litert": package_version("ai-edge-litert"),
        "tensorflow": package_version("tensorflow"),
        "riscv64-unknown-elf-gcc": os.environ["GCC_VERSION"],
        "spike": {
            "version": os.environ["SPIKE_VERSION"],
            "source_revision": os.environ["SPIKE_REVISION"],
            "source_branch": os.environ["SPIKE_BRANCH"],
            "executable": os.environ["SPIKE_BIN"],
        },
    },
}

with open(metadata_file, "w", encoding="utf-8") as stream:
    json.dump(metadata, stream, indent=2)
    stream.write("\n")
PY

printf 'Frozen ResNet-18 model in %s\n' "${source_dir}"
printf 'Model SHA-256: %s\n' "${model_sha256}"
printf 'README SHA-256: %s\n' "${readme_sha256}"
printf 'Metadata: %s\n' "${metadata_file}"
