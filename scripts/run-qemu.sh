#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EFI_BIN="$ROOT/dist/cranos.efi"
IMG_PATH="$ROOT/dist/cranos-usb.img"
MODEL_PATH="${1:-}"
TIMEOUT_SECS="${CRANOS_QEMU_TIMEOUT:-25}"
RAM_MB="${CRANOS_QEMU_RAM:-6144}"
CPUS="${CRANOS_QEMU_CPUS:-6}"
SCREENSHOT_WELCOME="${CRANOS_SCREENSHOT_WELCOME:-}"
SCREENSHOT_FINAL="${CRANOS_SCREENSHOT:-}"
AUTO_PROMPT="${CRANOS_AUTO_PROMPT:-}"

if [[ ! -f "$EFI_BIN" ]]; then
    echo "Missing $EFI_BIN; run scripts/build.sh first" >&2
    exit 1
fi

if [[ -n "$MODEL_PATH" && -f "$MODEL_PATH" ]]; then
    python3 "$ROOT/scripts/make_usb_image.py" "$EFI_BIN" "$IMG_PATH" "$MODEL_PATH"
else
    python3 "$ROOT/scripts/make_usb_image.py" "$EFI_BIN" "$IMG_PATH"
fi

OVMF_CODE="/usr/share/edk2/x64/OVMF_CODE.4m.fd"
OVMF_VARS_SRC="/usr/share/edk2/x64/OVMF_VARS.4m.fd"
OVMF_VARS="$ROOT/dist/OVMF_VARS.fd"
cp "$OVMF_VARS_SRC" "$OVMF_VARS"

KVM_ARGS=()
if [[ -r /dev/kvm && -w /dev/kvm ]]; then
    KVM_ARGS=(-enable-kvm -cpu host)
else
    KVM_ARGS=(-cpu max)
fi

MONITOR_SOCK="$ROOT/dist/qemu-mon.sock"
SERIAL_SOCK="$ROOT/dist/qemu-ser.sock"
rm -f "$MONITOR_SOCK" "$SERIAL_SOCK"

# Drive QEMU via Python controller so we can log serial live, wait for
# [CRANOS_READY], capture welcome screenshot, send a real prompt over COM1,
# wait for [llm] Done:, and capture the final Cranpose chat screenshot.
python3 - "$OVMF_CODE" "$OVMF_VARS" "$IMG_PATH" "$MONITOR_SOCK" "$SERIAL_SOCK" \
    "$TIMEOUT_SECS" "$RAM_MB" "$CPUS" "$SCREENSHOT_WELCOME" "$SCREENSHOT_FINAL" "$AUTO_PROMPT" \
    "${KVM_ARGS[@]}" <<'PY'
import os, select, socket, subprocess, sys, time

(
    ovmf_code, ovmf_vars, img_path, mon_sock, ser_sock,
    timeout_s, ram_mb, cpus, shot_welcome, shot_final, auto_prompt,
    *kvm_args
) = sys.argv[1:]
timeout_s = float(timeout_s)

cmd = [
    "qemu-system-x86_64",
    "-machine", "q35",
    *kvm_args,
    "-smp", cpus,
    "-m", ram_mb,
    "-drive", f"if=pflash,format=raw,readonly=on,file={ovmf_code}",
    "-drive", f"if=pflash,format=raw,file={ovmf_vars}",
    "-drive", f"id=bootdisk,if=none,format=raw,file={img_path}",
    "-device", "nvme,drive=bootdisk,serial=CRANOS01",
    "-device", "qemu-xhci",
    "-device", "usb-kbd",
    "-device", "usb-mouse",
    "-vga", "std",
    "-display", "none",
    "-serial", f"unix:{ser_sock},server,nowait",
    "-monitor", f"unix:{mon_sock},server,nowait",
    "-no-reboot",
]

proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def connect_sock(path: str):
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if os.path.exists(path):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(path)
                return s
            except OSError:
                pass
        time.sleep(0.05)
    raise RuntimeError(f"Failed to connect to {path}")

def screendump(mon: socket.socket, path: str):
    if not path:
        return
    mon.sendall(f"screendump {path}\n".encode())
    time.sleep(0.4)

ser = connect_sock(ser_sock)
mon = connect_sock(mon_sock)
ser.setblocking(False)

buf = ""
start = time.time()
sent_prompt = False
finished = False

try:
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            break
        r, _, _ = select.select([ser], [], [], 0.1)
        if r:
            chunk = ser.recv(4096)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            buf += text

            if not sent_prompt and "[CRANOS_READY]" in buf:
                time.sleep(0.4)
                screendump(mon, shot_welcome)
                if auto_prompt:
                    sent_prompt = True
                    # Type slowly like a real user so serial FIFO never drops chars
                    for ch in auto_prompt:
                        ser.sendall(ch.encode("utf-8"))
                        time.sleep(0.008)
                    ser.sendall(b"\r")
                else:
                    time.sleep(0.3)
                    screendump(mon, shot_final)
                    finished = True
                    break

            if sent_prompt and "[llm] Done:" in buf and "\n" in buf.split("[llm] Done:", 1)[1]:
                time.sleep(0.5)
                screendump(mon, shot_final)
                finished = True
                break
    if not finished:
        screendump(mon, shot_final)
finally:
    try:
        mon.sendall(b"quit\n")
        time.sleep(0.2)
    except OSError:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
PY
