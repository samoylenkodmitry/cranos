# CranOS

CranOS is a bare-metal x86_64 operating system in Rust and assembly. It boots directly from UEFI into a Cranpose UI and runs a multi-core AVX2/AVX-VNNI Qwen3 LLM engine without a Linux kernel or Linux userspace.



<img width="4080" height="3072" alt="PXL_20261126_105220330 MP" src="https://github.com/user-attachments/assets/4428a3a0-dd12-422d-b397-8b6b81de3f61" />



https://github.com/user-attachments/assets/ff6946e1-d90a-491f-a782-d7b3f6d4dc53





## Architecture

- **`crates/cranos-boot`** (`x86_64-unknown-uefi`): Pure-Rust UEFI Stage 1 loader. It picks a linear 32-bit GOP framebuffer (preferring `1920x1080`), locates ACPI RSDP, reads `model.gguf` from FAT32 (`SimpleFileSystem`) or ext4 partitions (`BlockIo` extent reader with `IoAlign` handling) into 2 MiB-aligned physical RAM, loads the embedded higher-half kernel ELF, builds 1 GiB/2 MiB 4-level page tables below 4 GiB, exits UEFI Boot Services, enables SSE and AVX, and jumps to the kernel.
- **`crates/cranos-kernel`** (`x86_64-cranos.json`): Higher-half x86_64 kernel running Cranpose directly on hardware:
  - Custom GDT, TSS (IST double-fault stack), and 256-entry IDT with automatic recovery reboot on fatal faults.
  - 512 GiB identity mapping in 1 GiB huge pages plus PAT Write-Combining remapping for the GOP framebuffer.
  - Calibrated RDTSC nanosecond clock (`CPUID` 0x15/0x16 with PIT channel 2 fallback) backing Cranpose `web-time`.
  - ACPI MADT parser and 16/32/64-bit assembly AP trampoline (`INIT`/`SIPI`). Secondary SMT threads park in `cli; hlt`. Physical cores claim dense worker slots behind a lock-free atomic spin barrier.
  - Multi-controller PCI + USB xHCI driver with BIOS handoff (`USBLEGSUP`), scratchpad allocation, composite single-command endpoint configuration, live root-hub hot-plug probing, boot keyboard, relative mouse, and 5-byte absolute pointer decoding.
  - PS/2 i8042 keyboard polling (without AUX/MUX probes) and probed 16550 COM1 serial input.
  - Cranpose `Framebuffer` (`PixelsRenderer` dirty-rect CPU blitter) rendering the chat UI and live hardware telemetry.
- **`crates/cranos-llm`** (`#![no_std]` + `alloc`): Zero-copy GGUF v2/v3 parser, byte-level BPE tokenizer, Qwen3/Qwen2/Llama-3 transformer engine (GQA, QK-Norm, NeoX/interleaved RoPE, SwiGLU, tied embeddings, F16 KV cache), and 4-row unrolled AVX2 + FMA + F16C + AVX-VNNI (`vpdpbusd`) kernels for `Q8_0`, `Q4_K`, `Q6_K`, `Q5_K`, `Q4_0`, `F16`, and `F32`.

## Verified Hardware Results

- **AMD Ryzen 9 3950X laptop (`samarch`, bare metal)**: 32 logical threads detected, 16 physical Zen 2 cores online (`1326 ns` barrier), `62910 MiB` usable RAM, `1920x1080` PAT WC framebuffer, 3 xHCI controllers and 3 USB HID interfaces bound, `Qwen3-4B-Q4_K_M.gguf` (`2375 MiB`) loaded from NVMe ext4 into 1 GiB huge pages and parsed in `221 ms`, total kernel boot `1245 ms`.
- **Intel Core i5-12400 (`samarch-1`, QEMU+OVMF+KVM)**: `Qwen3-0.6B-Q8_0.gguf` at `58.3 tok/s` (`36.9 GB/s`) and `Qwen3-4B-Q4_K_M.gguf` at `14.9 tok/s` (`37.1 GB/s`), `~303-357 ns` barrier.

## Prerequisites and Build

CranOS builds against the `feat/no-std-framebuffer` branch of `Cranpose` checked out as a sibling directory `../Cranpose-nostd` alongside this repository.

```bash
# Build dist/cranos.efi (embeds cranos-kernel inside cranos-boot)
bash scripts/build.sh

# Boot in QEMU + OVMF + KVM with a Qwen3 GGUF model
CRANOS_QEMU_RAM=8192 CRANOS_QEMU_CPUS=6 \
  bash scripts/run-qemu.sh /path/to/Qwen3-4B-Q4_K_M.gguf

# Verify ext4 NVMe BlockIo model loading in QEMU + OVMF + KVM
python3 scripts/test_ext4_boot.py /path/to/Qwen3-0.6B-Q8_0.gguf
```

## Bootable USB Flash Drive

```bash
# Create a GPT + FAT32 UEFI USB image containing /EFI/BOOT/BOOTX64.EFI and /model.gguf
python3 scripts/make_usb_image.py dist/cranos.efi dist/cranos-usb.img /path/to/Qwen3-4B-Q4_K_M.gguf

# Write to a USB flash drive (replace /dev/sdX with the removable disk)
sudo dd if=dist/cranos-usb.img of=/dev/sdX bs=4M status=progress conv=fsync
```

## Independent UEFI Entry on Arch UKI Hosts

`scripts/install-samarch-uefi.sh` installs `cranos.efi` into `/boot/efi/EFI/cranos/cranos.efi`, places `model.gguf` in `/cranos/model.gguf` (or `/media/data/cranos/model.gguf` when `/media/data` is mounted), and registers an independent `efibootmgr` entry while keeping the existing first `BootOrder` entry and `/boot/efi/EFI/Linux/` untouched.

```bash
sudo bash scripts/install-samarch-uefi.sh dist/cranos.efi /path/to/Qwen3-4B-Q4_K_M.gguf
```

Inside CranOS, press `F1` to toggle thinking mode, `F2` to run the speed benchmark, `F3` to clear chat, and `F10` (or `Ctrl+Alt+Del` or the `Reboot` button) to reboot cleanly via PCI `0xCF9`.
