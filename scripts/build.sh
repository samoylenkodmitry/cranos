#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE="${CRANOS_PROFILE:-release}"
CARGO_FLAG="--release"
TARGET_SUBDIR="release"
if [[ "$PROFILE" == "dev" ]]; then
    CARGO_FLAG=""
    TARGET_SUBDIR="debug"
fi

echo "==> [1/2] Building cranos-kernel (x86_64-cranos.json, $PROFILE)..."
RUSTFLAGS="-C link-arg=-T$ROOT/crates/cranos-kernel/linker.ld" \
    cargo +nightly build \
    -p cranos-kernel \
    $CARGO_FLAG \
    --target "$ROOT/x86_64-cranos.json" \
    -Z json-target-spec \
    -Z build-std=std,panic_abort \
    -Z build-std-features=compiler-builtins-mem \
    --manifest-path "$ROOT/Cargo.toml"

KERNEL_ELF="$ROOT/target/x86_64-cranos/$TARGET_SUBDIR/cranos-kernel"
ls -lh "$KERNEL_ELF"

echo "==> [2/2] Building cranos-boot (x86_64-unknown-uefi, $PROFILE)..."
CRANOS_KERNEL_ELF="$KERNEL_ELF" \
    cargo +nightly build \
    -p cranos-boot \
    $CARGO_FLAG \
    --target x86_64-unknown-uefi \
    --manifest-path "$ROOT/Cargo.toml"

mkdir -p "$ROOT/dist"
cp "$ROOT/target/x86_64-unknown-uefi/$TARGET_SUBDIR/cranos-boot.efi" "$ROOT/dist/cranos.efi"
ls -lh "$ROOT/dist/cranos.efi"
echo "==> Built $ROOT/dist/cranos.efi"
