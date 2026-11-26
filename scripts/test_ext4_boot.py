#!/usr/bin/env python3
"""Integration test verifying CranOS UEFI boot from an ESP-only disk while
loading /cranos/model.gguf from a separate ext4 GPT NVMe disk via cranos-boot's
ext4-over-BlockIo extent reader (the exact layout used on samarch-1)."""

import os
import shutil
import struct
import subprocess
import sys
import uuid
import zlib

SECTOR = 512


def build_gpt_ext4_disk(model_path: str, out_disk: str, stage_root: str) -> None:
    if os.path.exists(stage_root):
        shutil.rmtree(stage_root)
    os.makedirs(os.path.join(stage_root, "cranos"))
    dst_model = os.path.join(stage_root, "cranos", "model.gguf")
    try:
        os.link(model_path, dst_model)
    except OSError:
        shutil.copyfile(model_path, dst_model)

    part_img = out_disk + ".part"
    for p in (part_img, out_disk):
        if os.path.exists(p):
            os.remove(p)

    model_mb = (os.path.getsize(model_path) // (1024 * 1024)) + 96
    subprocess.check_call(["truncate", "-s", f"{model_mb}M", part_img])
    subprocess.check_call(["mkfs.ext4", "-q", "-F", "-d", stage_root, part_img])

    part_sectors = (model_mb * 1024 * 1024) // SECTOR
    start_lba = 2048
    end_lba = start_lba + part_sectors - 1
    total_lbas = end_lba + 2048

    mbr = bytearray(SECTOR)
    mbr[446:462] = struct.pack(
        "<BBBBBBBBII", 0, 0, 2, 0, 0xEE, 0xFF, 0xFF, 0xFF, 1, total_lbas - 1
    )
    mbr[510:512] = b"\x55\xAA"

    linux_guid = uuid.UUID("0FC63DAF-8483-4772-8E79-3D69D8477DE4").bytes_le
    part_guid = uuid.uuid5(uuid.NAMESPACE_DNS, "cranos.ext4.test").bytes_le
    disk_guid = uuid.uuid5(uuid.NAMESPACE_DNS, "cranos.ext4.disk").bytes_le
    entries = bytearray(128 * 128)
    entries[0:128] = (
        linux_guid
        + part_guid
        + struct.pack("<QQQ", start_lba, end_lba, 0)
        + "data".encode("utf-16le").ljust(72, b"\x00")
    )
    ecrc = zlib.crc32(entries) & 0xFFFFFFFF

    def gpt_hdr(my_lba: int, alt_lba: int, e_lba: int) -> bytes:
        h = bytearray(SECTOR)
        struct.pack_into(
            "<8sIIIIQQQQ16sQIII",
            h,
            0,
            b"EFI PART",
            0x10000,
            92,
            0,
            0,
            my_lba,
            alt_lba,
            34,
            total_lbas - 34,
            disk_guid,
            e_lba,
            128,
            128,
            ecrc,
        )
        struct.pack_into("<I", h, 16, zlib.crc32(h[:92]) & 0xFFFFFFFF)
        return bytes(h)

    with open(out_disk, "wb") as f:
        f.truncate(total_lbas * SECTOR)
        f.seek(0)
        f.write(mbr)
        f.seek(512)
        f.write(gpt_hdr(1, total_lbas - 1, 2))
        f.seek(1024)
        f.write(entries)
        f.seek((total_lbas - 33) * SECTOR)
        f.write(entries)
        f.seek((total_lbas - 1) * SECTOR)
        f.write(gpt_hdr(total_lbas - 1, 1, total_lbas - 33))
        f.seek(start_lba * SECTOR)
        with open(part_img, "rb") as src:
            shutil.copyfileobj(src, f, 8 * 1024 * 1024)

    os.remove(part_img)
    shutil.rmtree(stage_root)


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    efi_bin = os.path.join(root, "dist", "cranos.efi")
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/media/data/cranos/Qwen3-0.6B-Q8_0.gguf"
    esp_img = os.path.join(root, "dist", "test-esp-only.img")
    ext4_disk = os.path.join(root, "dist", "test-ext4-disk.img")
    ovmf_vars = os.path.join(root, "dist", "OVMF_VARS_ext4.fd")
    stage_root = os.path.join(os.path.dirname(model_path), ".cranos-ext4-stage")

    subprocess.check_call(["python3", os.path.join(root, "scripts", "make_usb_image.py"), efi_bin, esp_img])
    build_gpt_ext4_disk(model_path, ext4_disk, stage_root)

    shutil.copyfile("/usr/share/edk2/x64/OVMF_VARS.4m.fd", ovmf_vars)
    cmd = [
        "timeout", "15s",
        "qemu-system-x86_64", "-machine", "q35", "-enable-kvm", "-cpu", "host",
        "-smp", "6", "-m", "4096",
        "-drive", "if=pflash,format=raw,readonly=on,file=/usr/share/edk2/x64/OVMF_CODE.4m.fd",
        "-drive", f"if=pflash,format=raw,file={ovmf_vars}",
        "-drive", f"id=esp,if=none,format=raw,file={esp_img}",
        "-device", "nvme,drive=esp,serial=ESP001",
        "-drive", f"id=ext4d,if=none,format=raw,file={ext4_disk}",
        "-device", "nvme,drive=ext4d,serial=DATA01",
        "-vga", "std", "-display", "none", "-serial", "stdio", "-no-reboot",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        assert "Loaded ext4:/cranos/model.gguf" in res.stdout, "ext4 model load did not occur"
        assert "[CRANOS_READY]" in res.stdout and "model_loaded=true" in res.stdout, "CranOS did not load ext4 model"
        print("EXT4_NVME_BOOT_VERIFIED_OK")
    finally:
        for p in (esp_img, ext4_disk, ovmf_vars):
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    main()
