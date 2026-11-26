#!/usr/bin/env bash
# Safely installs CranOS as an independent UEFI boot entry on an Arch UKI host
# (such as samarch).
#
# Safety guarantees for Arch UKI setups:
#   - NEVER touches /boot/efi/EFI/Linux/* or /boot/vmlinuz-*.
#   - NEVER removes any existing UEFI boot entry.
#   - NEVER changes the #1 default BootOrder entry (e.g. Arch UKI cachyos-bore).
#   - Places CranOS in its own isolated /boot/efi/EFI/cranos/cranos.efi path.
#   - Places the LLM GGUF in /cranos/model.gguf on the root ext4 partition (or
#     /media/data/cranos/model.gguf when /media/data is a separate ext4 mount),
#     so the 512 MiB ESP only stores the ~7 MiB cranos.efi binary.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EFI_SRC="${1:-$ROOT/dist/cranos.efi}"
MODEL_SRC="${2:-}"
ESP_MOUNT="/boot/efi"
EFI_DIR="$ESP_MOUNT/EFI/cranos"
EFI_DST="$EFI_DIR/cranos.efi"
EFI_REL='\EFI\cranos\cranos.efi'
LABEL="CranOS Bare-Metal AI"

if [[ $EUID -ne 0 ]]; then
    echo "Run with sudo: sudo bash $0 [cranos.efi] [model.gguf]" >&2
    exit 1
fi

if [[ ! -f "$EFI_SRC" ]]; then
    echo "Missing EFI binary: $EFI_SRC" >&2
    exit 1
fi

if [[ ! -d "$ESP_MOUNT/EFI" ]]; then
    echo "ESP is not mounted at $ESP_MOUNT" >&2
    exit 1
fi

esp_src="$(findmnt -no SOURCE "$ESP_MOUNT" | tail -n1)"
disk="/dev/$(lsblk -no PKNAME "$esp_src")"
part="$(lsblk -no PARTN "$esp_src")"

echo "==> Installing CranOS EFI binary into isolated $EFI_DST (leaving EFI/Linux untouched)..."
mkdir -p "$EFI_DIR"
install -m 0644 "$EFI_SRC" "$EFI_DST"

if [[ -n "$MODEL_SRC" && -f "$MODEL_SRC" ]]; then
    if mountpoint -q /media/data 2>/dev/null; then
        MODEL_DIR="/media/data/cranos"
    else
        MODEL_DIR="/cranos"
    fi
    MODEL_DST="$MODEL_DIR/model.gguf"
    mkdir -p "$MODEL_DIR"
    if [[ "$(realpath "$MODEL_SRC")" != "$(realpath -m "$MODEL_DST")" ]]; then
        ln -f "$(realpath "$MODEL_SRC")" "$MODEL_DST" 2>/dev/null \
            || install -m 0644 "$MODEL_SRC" "$MODEL_DST"
    fi
    echo "==> Model ready at $MODEL_DST ($(du -h "$MODEL_DST" | awk '{print $1}'))"
fi

# Save current BootOrder so the existing #1 default entry stays first.
ORIG_ORDER="$(efibootmgr | awk '/^BootOrder:/ { print $2 }')"

if efibootmgr -v | grep -Fqi "$EFI_REL"; then
    echo "==> UEFI boot entry for $EFI_REL already exists."
else
    echo "==> Creating independent UEFI boot entry '$LABEL' on $disk partition $part..."
    efibootmgr -c -d "$disk" -p "$part" -L "$LABEL" -l "$EFI_REL" >/dev/null
    CRANOS_NUM="$(efibootmgr -v | awk -F'[*\t]' 'index(tolower($0), "\\efi\\cranos\\cranos.efi") > 0 { match($1, /Boot([0-9A-F]+)/, m); print m[1] }' | head -1)"
    if [[ -n "$ORIG_ORDER" && -n "$CRANOS_NUM" ]]; then
        CLEAN_ORIG="$(echo "$ORIG_ORDER" | tr ',' '\n' | grep -v "^${CRANOS_NUM}$" | tr '\n' ',' | sed 's/,$//' || true)"
        if [[ -n "$CLEAN_ORIG" ]]; then
            efibootmgr -o "${CLEAN_ORIG},${CRANOS_NUM}" >/dev/null
        fi
    fi
fi

echo ""
echo "==> Current UEFI Boot Entries (default #1 preserved):"
efibootmgr
echo ""
CRANOS_NUM="$(efibootmgr -v | awk -F'[*\t]' 'index(tolower($0), "\\efi\\cranos\\cranos.efi") > 0 { match($1, /Boot([0-9A-F]+)/, m); print m[1] }' | head -1)"
echo "To boot CranOS once on next reboot without changing default boot order:"
echo "  sudo efibootmgr --bootnext ${CRANOS_NUM:-XXXX} && sudo reboot"
echo "Or press F7 / F12 at BIOS POST to pick '$LABEL' from the UEFI Boot Menu."
