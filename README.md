# CranOS

Bare-metal x86_64 operating system written from scratch in Rust and assembly, booting directly through UEFI into a Cranpose UI and a multi-core AVX2/AVX-VNNI Qwen3 LLM engine with zero Linux kernel or userspace overhead.

## Architecture

- **`crates/cranos-boot`** (`x86_64-unknown-uefi`): Pure-Rust UEFI Stage 1 loader. Selects a linear 32-bit GOP framebuffer, locates ACPI RSDP, loads `model.gguf` from FAT32 (`SimpleFileSystem`) or ext4 partitions (`BlockIo` extent reader) into 2 MiB-aligned physical RAM, loads the embedded higher-half ELF kernel, builds 1 GiB/2 MiB huge-page 4-level page tables, exits UEFI Boot Services, enables SSE/AVX, and jumps to the kernel.
- **`crates/cranos-kernel`** (`x86_64-cranos.json`): Higher-half x86_64 bare-metal kernel with hardware SSE2/AVX2/AVX-VNNI:
  - Custom GDT, TSS (IST double-fault stack), and 256-entry IDT with assembly exception dumps.
  - 1 GiB huge-page identity mapping for RAM and LLM weights (zero TLB misses) plus PAT Write-Combining remapping for the GOP framebuffer.
  - Calibrated RDTSC nanosecond clock (`CPUID` 0x15/0x16 + PIT fallback) feeding Cranpose's `web-time` shim.
  - ACPI MADT parser + 16-bit/32-bit/64-bit assembly AP trampoline (`INIT`/`SIPI`) bringing up all physical P-cores with a lock-free atomic spin barrier (~400 ns per barrier).
  - Bare-metal PCI + USB xHCI controller driver (with BIOS-to-OS `USBLEGSUP` handoff and scratchpad buffers for Intel 600-series) binding USB HID boot keyboards and mice, plus PS/2 i8042 and COM1 serial input.
  - Real Cranpose `Framebuffer` (`PixelsRenderer` dirty-rect CPU blitting) running the interactive chat and live telemetry dashboard.
- **`crates/cranos-llm`** (`#![no_std]` + `alloc`): Zero-copy GGUF v2/v3 parser, byte-level BPE tokenizer, Qwen3/Qwen2/Llama-3 transformer engine (GQA, QK-Norm, NeoX/interleaved RoPE, SwiGLU, F16 KV cache), and 4-row unrolled AVX2 + FMA + F16C + AVX-VNNI (`vpdpbusd`) kernels for `Q8_0`, `Q4_K`, `Q6_K`, `Q5_K`, `Q4_0`, `F16`, and `F32`.

## Build and QEMU Test

```bash
# Build dist/cranos.efi (embeds cranos-kernel inside cranos-boot)
bash scripts/build.sh

# Boot in QEMU + OVMF + KVM with a Qwen3 GGUF model
CRANOS_QEMU_RAM=8192 CRANOS_QEMU_CPUS=6 \
  bash scripts/run-qemu.sh /media/data/cranos/Qwen3-4B-Q4_K_M.gguf
```

## Bootable USB Flash Drive

```bash
# Create a GPT + FAT32 UEFI USB image containing /EFI/BOOT/BOOTX64.EFI and /model.gguf
python3 scripts/make_usb_image.py dist/cranos.efi dist/cranos-usb.img /media/data/cranos/Qwen3-4B-Q4_K_M.gguf

# Write to a USB flash drive (replace /dev/sdX with the actual removable disk)
sudo dd if=dist/cranos-usb.img of=/dev/sdX bs=4M status=progress conv=fsync
```

## Independent UEFI Entry on `samarch-1`

`scripts/install-samarch-uefi.sh` copies `cranos.efi` into `/boot/efi/EFI/cranos/cranos.efi`, links `/media/data/cranos/model.gguf` on `nvme0n1p2` (read by `cranos-boot`'s ext4 loader without taking space on `/boot/efi`), and creates an independent `efibootmgr` entry while keeping `Arch UKI cachyos-bore` as the first entry in `BootOrder` and leaving `/usr/local/sbin/direct-uki-sync` and `/boot/efi/EFI/Linux/` untouched.

```bash
sudo bash scripts/install-samarch-uefi.sh dist/cranos.efi /media/data/cranos/Qwen3-4B-Q4_K_M.gguf
```
