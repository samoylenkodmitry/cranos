#!/usr/bin/env python3
"""Build a UEFI-bootable GPT + FAT32 disk/USB image for CranOS without root.

Creates a standards-compliant GPT disk with a single FAT32 EFI System Partition
containing /EFI/BOOT/BOOTX64.EFI and (optionally) /model.gguf with VFAT LFN.
The resulting raw image boots directly in QEMU+OVMF+KVM and can be written to
any USB flash drive with `dd`.
"""

import os
import struct
import sys
import uuid
import zlib

SECTOR = 512
PART_START_LBA = 2048  # 1 MiB aligned


def lfn_checksum(short_name_11: bytes) -> int:
    s = 0
    for b in short_name_11:
        s = (((s & 1) << 7) + (s >> 1) + b) & 0xFF
    return s


def encode_lfn_entries(long_name: str, short_name_11: bytes) -> bytes:
    chk = lfn_checksum(short_name_11)
    # UTF-16LE code units + NUL + 0xFFFF padding to multiple of 13
    units = [ord(c) for c in long_name] + [0x0000]
    while len(units) % 13 != 0:
        units.append(0xFFFF)
    chunks = [units[i : i + 13] for i in range(0, len(units), 13)]
    out = bytearray()
    n = len(chunks)
    for idx in range(n - 1, -1, -1):
        seq = (idx + 1) | (0x40 if idx == n - 1 else 0)
        ch = chunks[idx]
        part1 = b"".join(struct.pack("<H", u) for u in ch[0:5])
        part2 = b"".join(struct.pack("<H", u) for u in ch[5:11])
        part3 = b"".join(struct.pack("<H", u) for u in ch[11:13])
        entry = (
            struct.pack("<B", seq)
            + part1
            + struct.pack("<BBB", 0x0F, 0x00, chk)
            + part2
            + struct.pack("<H", 0x0000)
            + part3
        )
        assert len(entry) == 32
        out += entry
    return bytes(out)


def make_dir_entry(short_11: bytes, attr: int, first_cluster: int, size: int) -> bytes:
    assert len(short_11) == 11
    hi = (first_cluster >> 16) & 0xFFFF
    lo = first_cluster & 0xFFFF
    return struct.pack(
        "<11sBBBHHHHHHHL",
        short_11,
        attr,
        0,  # NTRes
        0,  # CrtTimeTenth
        0,  # CrtTime
        0x5921,  # CrtDate
        0x5921,  # LstAccDate
        hi,
        0,  # WrtTime
        0x5921,  # WrtDate
        lo,
        size,
    )


def build_image(efi_path: str, out_path: str, model_path: str | None) -> None:
    efi_size = os.path.getsize(efi_path)
    model_size = os.path.getsize(model_path) if model_path else 0
    if model_size >= 0xFFFF_FFFF:
        raise ValueError("FAT32 max single file size is 4 GiB - 1")

    total_payload = efi_size + model_size
    # Choose cluster size so total_clusters >= 66000 (strict FAT32 threshold = 65525).
    if total_payload < 1200 * 1024 * 1024:
        spc = 8  # 4 KiB clusters
        min_part_bytes = 300 * 1024 * 1024
    else:
        spc = 32  # 16 KiB clusters
        min_part_bytes = 1200 * 1024 * 1024

    cluster_bytes = spc * SECTOR
    # Add generous overhead for FATs + alignment + slack
    part_bytes = max(min_part_bytes, total_payload + 96 * 1024 * 1024)
    part_sectors = ((part_bytes + cluster_bytes - 1) // cluster_bytes) * spc

    reserved_sectors = 32
    num_fats = 2
    # Iteratively compute FAT size in sectors
    data_clusters = (part_sectors - reserved_sectors) // spc
    fat_sectors = (data_clusters * 4 + SECTOR - 1) // SECTOR
    data_sectors = part_sectors - reserved_sectors - num_fats * fat_sectors
    total_clusters = data_sectors // spc
    assert total_clusters >= 65525, f"FAT32 requires >=65525 clusters, got {total_clusters}"

    part_end_lba = PART_START_LBA + part_sectors - 1
    total_lbas = part_end_lba + 2048  # Room for backup GPT + alignment

    print(
        f"[make-usb] Building {out_path}: {total_lbas * SECTOR / (1024*1024):.1f} MiB "
        f"(clusters={total_clusters}, cluster={cluster_bytes}B, efi={efi_size//1024}KiB, "
        f"model={model_size//(1024*1024)}MiB)"
    )

    # Cluster allocation:
    # 2 = root dir (/)
    # 3 = /EFI
    # 4 = /EFI/BOOT
    # 5.. = /EFI/BOOT/BOOTX64.EFI
    # next.. = /model.gguf (if present)
    next_cluster = 5

    efi_clusters = max(1, (efi_size + cluster_bytes - 1) // cluster_bytes)
    efi_start = next_cluster
    next_cluster += efi_clusters

    model_start = 0
    model_clusters = 0
    if model_path:
        model_clusters = max(1, (model_size + cluster_bytes - 1) // cluster_bytes)
        model_start = next_cluster
        next_cluster += model_clusters

    assert next_cluster < total_clusters + 2

    # Build FAT table in memory
    fat = bytearray(fat_sectors * SECTOR)

    def set_fat(cl: int, val: int) -> None:
        struct.pack_into("<I", fat, cl * 4, val & 0x0FFF_FFFF)

    set_fat(0, 0x0FFF_FFF8)
    set_fat(1, 0x0FFF_FFFF)
    set_fat(2, 0x0FFF_FFFF)  # /
    set_fat(3, 0x0FFF_FFFF)  # /EFI
    set_fat(4, 0x0FFF_FFFF)  # /EFI/BOOT

    for i in range(efi_clusters):
        set_fat(efi_start + i, 0x0FFF_FFFF if i + 1 == efi_clusters else efi_start + i + 1)

    if model_path:
        for i in range(model_clusters):
            set_fat(
                model_start + i,
                0x0FFF_FFFF if i + 1 == model_clusters else model_start + i + 1,
            )

    # Build directories (1 cluster each)
    root_dir = bytearray(cluster_bytes)
    off = 0
    # Volume label entry
    root_dir[off : off + 32] = make_dir_entry(b"CRANOS     ", 0x08, 0, 0)
    off += 32
    # EFI directory
    root_dir[off : off + 32] = make_dir_entry(b"EFI        ", 0x10, 3, 0)
    off += 32
    if model_path:
        short_model = b"MODEL~1 GGU"
        lfn = encode_lfn_entries("model.gguf", short_model)
        root_dir[off : off + len(lfn)] = lfn
        off += len(lfn)
        root_dir[off : off + 32] = make_dir_entry(short_model, 0x20, model_start, model_size)
        off += 32

    efi_dir = bytearray(cluster_bytes)
    efi_dir[0:32] = make_dir_entry(b".          ", 0x10, 3, 0)
    efi_dir[32:64] = make_dir_entry(b"..         ", 0x10, 0, 0)
    efi_dir[64:96] = make_dir_entry(b"BOOT       ", 0x10, 4, 0)

    boot_dir = bytearray(cluster_bytes)
    boot_dir[0:32] = make_dir_entry(b".          ", 0x10, 4, 0)
    boot_dir[32:64] = make_dir_entry(b"..         ", 0x10, 3, 0)
    boot_dir[64:96] = make_dir_entry(b"BOOTX64 EFI", 0x20, efi_start, efi_size)

    # Build FAT32 BPB (sector 0 of partition)
    bpb = bytearray(SECTOR)
    bpb[0:3] = b"\xEB\x58\x90"
    bpb[3:11] = b"CRANOS  "
    struct.pack_into("<H", bpb, 11, SECTOR)
    bpb[13] = spc
    struct.pack_into("<H", bpb, 14, reserved_sectors)
    bpb[16] = num_fats
    struct.pack_into("<H", bpb, 17, 0)
    struct.pack_into("<H", bpb, 19, 0)
    bpb[21] = 0xF8
    struct.pack_into("<H", bpb, 22, 0)
    struct.pack_into("<H", bpb, 24, 63)
    struct.pack_into("<H", bpb, 26, 255)
    struct.pack_into("<I", bpb, 28, PART_START_LBA)
    struct.pack_into("<I", bpb, 32, part_sectors)
    struct.pack_into("<I", bpb, 36, fat_sectors)
    struct.pack_into("<H", bpb, 40, 0)
    struct.pack_into("<H", bpb, 42, 0)
    struct.pack_into("<I", bpb, 44, 2)  # Root cluster
    struct.pack_into("<H", bpb, 48, 1)  # FSInfo sector
    struct.pack_into("<H", bpb, 50, 6)  # Backup boot sector
    bpb[64] = 0x80
    bpb[66] = 0x29
    struct.pack_into("<I", bpb, 67, 0x4352_414E)
    bpb[71:82] = b"CRANOS     "
    bpb[82:90] = b"FAT32   "
    bpb[510:512] = b"\x55\xAA"

    # FSInfo sector (sector 1)
    fsinfo = bytearray(SECTOR)
    struct.pack_into("<I", fsinfo, 0, 0x4161_5252)
    struct.pack_into("<I", fsinfo, 484, 0x6141_7272)
    struct.pack_into("<I", fsinfo, 488, total_clusters - (next_cluster - 2))
    struct.pack_into("<I", fsinfo, 492, next_cluster)
    struct.pack_into("<I", fsinfo, 508, 0xAA55_0000)

    # Protective MBR (LBA 0)
    mbr = bytearray(SECTOR)
    mbr[446 : 446 + 16] = struct.pack(
        "<BBBBBBBBII",
        0x00,
        0x00,
        0x02,
        0x00,
        0xEE,  # GPT protective
        0xFF,
        0xFF,
        0xFF,
        1,
        min(total_lbas - 1, 0xFFFF_FFFF),
    )
    mbr[510:512] = b"\x55\xAA"

    # GPT Partition Entries (128 entries * 128 bytes = 16 KiB = 32 sectors)
    esp_type_guid = uuid.UUID("C12A7328-F81F-11D2-BA4B-00A0C93EC93B").bytes_le
    part_guid = uuid.uuid5(uuid.NAMESPACE_DNS, "cranos.esp.partition").bytes_le
    disk_guid = uuid.uuid5(uuid.NAMESPACE_DNS, "cranos.usb.disk").bytes_le

    part_entries = bytearray(128 * 128)
    name_utf16 = "CranOS EFI".encode("utf-16le").ljust(72, b"\x00")
    part_entries[0:128] = (
        esp_type_guid
        + part_guid
        + struct.pack("<QQQ", PART_START_LBA, part_end_lba, 0)
        + name_utf16
    )
    entries_crc = zlib.crc32(part_entries) & 0xFFFF_FFFF

    def make_gpt_header(my_lba: int, alt_lba: int, entries_lba: int) -> bytes:
        hdr = bytearray(SECTOR)
        struct.pack_into(
            "<8sIIIIQQQQ16sQIII",
            hdr,
            0,
            b"EFI PART",
            0x0001_0000,
            92,
            0,
            0,
            my_lba,
            alt_lba,
            34,
            total_lbas - 34,
            disk_guid,
            entries_lba,
            128,
            128,
            entries_crc,
        )
        hdr_crc = zlib.crc32(hdr[:92]) & 0xFFFF_FFFF
        struct.pack_into("<I", hdr, 16, hdr_crc)
        return bytes(hdr)

    primary_gpt = make_gpt_header(1, total_lbas - 1, 2)
    backup_gpt = make_gpt_header(total_lbas - 1, 1, total_lbas - 33)

    # Write disk image directly with sparse pre-allocation + streaming file copies
    part_byte_off = PART_START_LBA * SECTOR
    data_start_byte = part_byte_off + (reserved_sectors + num_fats * fat_sectors) * SECTOR

    def cluster_off(cl: int) -> int:
        return data_start_byte + (cl - 2) * cluster_bytes

    with open(out_path, "wb") as f:
        f.truncate(total_lbas * SECTOR)
        # GPT + Protective MBR
        f.seek(0)
        f.write(mbr)
        f.seek(1 * SECTOR)
        f.write(primary_gpt)
        f.seek(2 * SECTOR)
        f.write(part_entries)
        f.seek((total_lbas - 33) * SECTOR)
        f.write(part_entries)
        f.seek((total_lbas - 1) * SECTOR)
        f.write(backup_gpt)

        # FAT32 BPB, backup BPB, FSInfo, FAT1, FAT2
        f.seek(part_byte_off)
        f.write(bpb)
        f.seek(part_byte_off + 1 * SECTOR)
        f.write(fsinfo)
        f.seek(part_byte_off + 6 * SECTOR)
        f.write(bpb)
        f.seek(part_byte_off + reserved_sectors * SECTOR)
        f.write(fat)
        f.write(fat)

        # Directories
        f.seek(cluster_off(2))
        f.write(root_dir)
        f.seek(cluster_off(3))
        f.write(efi_dir)
        f.seek(cluster_off(4))
        f.write(boot_dir)

        # Copy BOOTX64.EFI
        f.seek(cluster_off(efi_start))
        with open(efi_path, "rb") as src:
            while chunk := src.read(4 * 1024 * 1024):
                f.write(chunk)

        # Copy model.gguf
        if model_path:
            f.seek(cluster_off(model_start))
            with open(model_path, "rb") as src:
                while chunk := src.read(8 * 1024 * 1024):
                    f.write(chunk)

    print(f"[make-usb] Wrote {out_path} successfully.")


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(f"Usage: {sys.argv[0]} <cranos.efi> <out.img> [model.gguf]", file=sys.stderr)
        sys.exit(1)
    build_image(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None)
