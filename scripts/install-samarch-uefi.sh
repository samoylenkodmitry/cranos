#!/usr/bin/env bash
# Safely installs CranOS as an independent UEFI boot entry on samarch-1.
#
# Safety guarantees for samarch-1's Arch UKI setup:
#   - NEVER touches /usr/local/sbin/direct-uki-sync or pacman hooks.
#   - NEVER touches /boot/efi/EFI/Linux/* or /boot/vmlinuz-*.
#   - NEVER changes the default BootOrder first entry (Arch UKI cachyos-bore).
#   - Places CranOS in its own isolated /boot/efi/EFI/cranos/cranos.efi path.
#   - Places the LLM GGUF in /media/data/cranos/model.gguf (nvme0n1p2 ext4)
#     so /boot/efi (512 MiB ESP) uses < 7 MiB.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EFI_SRC="${1:-$ROOT/dist/cranos.efi}"
MODEL_SRC="${2:-/media/data/cranos/Qwen3-4B-Q4_K_M.gguf}"
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

if [[ -f "$MODEL_SRC" ]]; then
    mkdir -p /media/data/cranos
    if [[ "$(realpath "$MODEL_SRC")" != "/media/data/cranos/model.gguf" ]]; then
        ln -sf "$(realpath "$MODEL_SRC")" /media/data/cranos/model.gguf
        # Note: our bare-metal ext4 reader opens regular files directly; copy or hardlink if symlink.
        rm -f /media/data/cranos/model.gguf
        ln "$(realpath "$MODEL_SRC")" /media/data/cranos/model.gguf 2>/dev/null \
            || cp -a "$MODEL_SRC" /media/data/cranos/model.gguf
    fi
    echo "==> Model ready at /media/data/cranos/model.gguf ($(du -h /media/data/cranos/model.gguf | awk '{print $1}'))"
fi

# Save current BootOrder so we guarantee Arch UKI remains the #1 default boot entry.
ORIG_ORDER="$(efibootmgr | awk '/^BootOrder:/ { print $2 }')"

if efibootmgr -v | grep -Fqi "$EFI_REL"; then
    echo "==> UEFI boot entry for $EFI_REL already exists."
else
    echo "==> Creating independent UEFI boot entry '$LABEL' on $disk partition $part..."
    efibootmgr -c -d "$disk" -p "$part" -L "$LABEL" -l "$EFI_REL" >/dev/null
    # Restore original default first entry while keeping CranOS appended at the end.
    CRANOS_NUM="$(efibootmgr -v | awk -F'[*\t]' 'tolower($0) ~ /\\efi\\cranos\\cranos\.efi/ { match($1, /Boot([0-9A-F]+)/, m); print m[1] }' | head -1)"
    if [[ -n "$ORIG_ORDER" && -n "$CRANOS_NUM" ]]; then
        CLEAN_ORIG="$(echo "$ORIG_ORDER" | tr ',' '\n' | grep -v "^${CRANOS_NUM}$" | tr '\n' ',' | sed 's/,$//')"
        efibootmgr -o "${CLEAN_ORIG},${CRANOS_NUM}" >/dev/null
    fi
fi

echo ""
echo "==> Current UEFI Boot Entries (Arch UKI remains default #1):"
efibootmgr
echo ""
echo "To boot CranOS once on next reboot without changing default boot order:"
CRANOS_NUM="$(efibootmgr -v | awk -F'[*\t]' 'tolower($0) ~ /\\efi\\cranos\\cranos\.efi/ { match($1, /Boot([0-9A-F]+)/, m); print m[1] }' | head -1)"
echo "  sudo efibootmgr --bootnext ${CRANOS_NUM:-XXXX} && sudo reboot"
echo "Or press F8 at ASUS D700MD POST to pick '$LABEL' from the UEFI Boot Menu."
