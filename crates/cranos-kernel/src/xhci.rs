use alloc::vec::Vec;
use core::arch::asm;
use core::sync::atomic::{Ordering, fence};

use crate::cpu;
use crate::input::{self, InputEvent};
use crate::kprintln;
use crate::mem;

const PCI_CONFIG_ADDR: u16 = 0xCF8;
const PCI_CONFIG_DATA: u16 = 0xCFC;

#[inline]
unsafe fn outl(port: u16, val: u32) {
    unsafe { asm!("out dx, eax", in("dx") port, in("eax") val, options(nomem, nostack, preserves_flags)) };
}

#[inline]
unsafe fn inl(port: u16) -> u32 {
    let val: u32;
    unsafe { asm!("in eax, dx", in("dx") port, out("eax") val, options(nomem, nostack, preserves_flags)) };
    val
}

fn pci_read32(bus: u8, dev: u8, func: u8, offset: u8) -> u32 {
    let addr = 0x8000_0000u32
        | ((bus as u32) << 16)
        | ((dev as u32) << 11)
        | ((func as u32) << 8)
        | ((offset as u32) & 0xFC);
    unsafe {
        outl(PCI_CONFIG_ADDR, addr);
        inl(PCI_CONFIG_DATA)
    }
}

fn pci_write32(bus: u8, dev: u8, func: u8, offset: u8, val: u32) {
    let addr = 0x8000_0000u32
        | ((bus as u32) << 16)
        | ((dev as u32) << 11)
        | ((func as u32) << 8)
        | ((offset as u32) & 0xFC);
    unsafe {
        outl(PCI_CONFIG_ADDR, addr);
        outl(PCI_CONFIG_DATA, val);
    }
}

#[derive(Clone, Copy)]
struct PciXhci {
    bus: u8,
    dev: u8,
    func: u8,
    vendor: u16,
    device: u16,
    bar0: u64,
}

fn find_xhci_pci() -> Option<PciXhci> {
    for bus in 0..=16u8 {
        for dev in 0..32u8 {
            for func in 0..8u8 {
                let id = pci_read32(bus, dev, func, 0x00);
                if id == 0xFFFF_FFFF || id == 0 {
                    if func == 0 {
                        break;
                    }
                    continue;
                }
                let class_reg = pci_read32(bus, dev, func, 0x08);
                let class = (class_reg >> 24) as u8;
                let subclass = ((class_reg >> 16) & 0xFF) as u8;
                let prog_if = ((class_reg >> 8) & 0xFF) as u8;
                if class == 0x0C && subclass == 0x03 && prog_if == 0x30 {
                    // Enable Memory Space + Bus Master
                    let cmd = pci_read32(bus, dev, func, 0x04);
                    pci_write32(bus, dev, func, 0x04, cmd | 0x06);

                    let bar0_lo = pci_read32(bus, dev, func, 0x10);
                    let is_64 = ((bar0_lo >> 1) & 0x3) == 2;
                    let bar0_hi = if is_64 {
                        pci_read32(bus, dev, func, 0x14)
                    } else {
                        0
                    };
                    let bar0 = ((bar0_hi as u64) << 32) | ((bar0_lo & !0xF) as u64);
                    if bar0 != 0 {
                        return Some(PciXhci {
                            bus,
                            dev,
                            func,
                            vendor: (id & 0xFFFF) as u16,
                            device: (id >> 16) as u16,
                            bar0,
                        });
                    }
                }
                if func == 0 {
                    let hdr = (pci_read32(bus, dev, 0, 0x0C) >> 16) as u8;
                    if (hdr & 0x80) == 0 {
                        break;
                    }
                }
            }
        }
    }
    None
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Trb {
    param: u64,
    status: u32,
    control: u32,
}

impl Trb {
    const ZERO: Self = Self {
        param: 0,
        status: 0,
        control: 0,
    };
}

#[repr(C, align(64))]
struct ErstEntry {
    base: u64,
    size: u32,
    reserved: u32,
}

const TRB_TYPE_NORMAL: u32 = 1;
const TRB_TYPE_SETUP: u32 = 2;
const TRB_TYPE_DATA: u32 = 3;
const TRB_TYPE_STATUS: u32 = 4;
const TRB_TYPE_LINK: u32 = 6;
const TRB_TYPE_ENABLE_SLOT: u32 = 9;
const TRB_TYPE_ADDRESS_DEVICE: u32 = 11;
const TRB_TYPE_CONFIGURE_EP: u32 = 12;
const TRB_TYPE_TRANSFER_EVENT: u32 = 32;
const TRB_TYPE_CMD_COMPLETION: u32 = 33;

const RING_LEN: usize = 64;
const EVENT_RING_LEN: usize = 256;

struct TrbRing {
    phys: u64,
    enq: usize,
    cycle: bool,
}

impl TrbRing {
    fn new() -> Option<Self> {
        let phys = mem::alloc_dma_zeroed(RING_LEN * 16, 4096)?;
        Some(Self {
            phys,
            enq: 0,
            cycle: true,
        })
    }

    fn push(&mut self, param: u64, status: u32, control_no_cycle: u32) -> u64 {
        let idx = self.enq;
        let ptr = unsafe { (self.phys as *mut Trb).add(idx) };
        let trb_addr = ptr as u64;
        let cyc = u32::from(self.cycle);
        unsafe {
            ptr.write_volatile(Trb {
                param,
                status,
                control: (control_no_cycle & !1) | cyc,
            });
        }
        self.enq += 1;
        if self.enq == RING_LEN - 1 {
            // Write Link TRB with Toggle Cycle bit (bit 1) wrapping to ring start.
            let link_ptr = unsafe { (self.phys as *mut Trb).add(RING_LEN - 1) };
            unsafe {
                link_ptr.write_volatile(Trb {
                    param: self.phys,
                    status: 0,
                    control: (TRB_TYPE_LINK << 10) | (1 << 1) | cyc,
                });
            }
            self.enq = 0;
            self.cycle = !self.cycle;
        }
        fence(Ordering::SeqCst);
        trb_addr
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HidKind {
    Keyboard,
    Mouse,
}

struct HidDevice {
    slot_id: u8,
    dci: u8,
    kind: HidKind,
    report_len: u16,
    ring: TrbRing,
    buf_phys: u64,
    prev_keys: [u8; 6],
    prev_mouse_btns: u8,
}

struct XhciController {
    cap_base: u64,
    op_base: u64,
    db_base: u64,
    rt_base: u64,
    ctx_size: usize, // 32 or 64
    max_ports: u8,
    dcbaa: u64,
    cmd_ring: TrbRing,
    evt_ring_phys: u64,
    evt_deq: usize,
    evt_cycle: bool,
    hid_devices: Vec<HidDevice>,
}

impl XhciController {
    #[inline]
    fn rd32(&self, addr: u64) -> u32 {
        unsafe { (addr as *const u32).read_volatile() }
    }

    #[inline]
    fn wr32(&self, addr: u64, val: u32) {
        unsafe { (addr as *mut u32).write_volatile(val) };
    }

    #[inline]
    fn wr64(&self, addr: u64, val: u64) {
        unsafe {
            (addr as *mut u32).write_volatile(val as u32);
            ((addr + 4) as *mut u32).write_volatile((val >> 32) as u32);
        }
    }

    #[inline]
    fn ring_doorbell(&self, slot: u8, target: u32) {
        fence(Ordering::SeqCst);
        self.wr32(self.db_base + (slot as u64) * 4, target);
    }

    fn portsc_addr(&self, port: u8) -> u64 {
        self.op_base + 0x400 + ((port as u64 - 1) * 0x10)
    }

    fn bios_handoff(&self, xecp_dwords: u32) {
        if xecp_dwords == 0 {
            return;
        }
        let mut off = (xecp_dwords as u64) * 4;
        for _ in 0..32 {
            let reg = self.rd32(self.cap_base + off);
            let cap_id = (reg & 0xFF) as u8;
            if cap_id == 1 {
                // USBLEGSUP: claim OS ownership (bit 24) and wait for BIOS (bit 16) to clear.
                self.wr32(self.cap_base + off, reg | (1 << 24));
                let deadline = cpu::now_nanos() + 500_000_000;
                while cpu::now_nanos() < deadline {
                    let cur = self.rd32(self.cap_base + off);
                    if (cur & (1 << 16)) == 0 {
                        break;
                    }
                    cpu::busy_wait_us(1000);
                }
                // Clear legacy SMI enables in USBLEGCTLSTS (offset + 4).
                let ctl = self.rd32(self.cap_base + off + 4);
                self.wr32(self.cap_base + off + 4, (ctl & 0x000E_0000) | 0xE000_0000);
                return;
            }
            let next = ((reg >> 8) & 0xFF) as u64;
            if next == 0 {
                break;
            }
            off += next * 4;
        }
    }

    fn pop_event(&mut self) -> Option<Trb> {
        let ptr = unsafe { (self.evt_ring_phys as *const Trb).add(self.evt_deq) };
        let trb = unsafe { ptr.read_volatile() };
        let cyc = (trb.control & 1) != 0;
        if cyc != self.evt_cycle {
            return None;
        }
        self.evt_deq += 1;
        if self.evt_deq == EVENT_RING_LEN {
            self.evt_deq = 0;
            self.evt_cycle = !self.evt_cycle;
        }
        let next_erdp = self.evt_ring_phys + (self.evt_deq as u64) * 16;
        // Write ERDP with EHB (bit 3) cleared/acked
        self.wr64(self.rt_base + 0x20 + 0x18, next_erdp | (1 << 3));
        Some(trb)
    }

    fn wait_event(&mut self, want_type: u32, timeout_ms: u64) -> Option<Trb> {
        let deadline = cpu::now_nanos() + timeout_ms * 1_000_000;
        while cpu::now_nanos() < deadline {
            if let Some(ev) = self.pop_event() {
                let ty = (ev.control >> 10) & 0x3F;
                if ty == want_type {
                    return Some(ev);
                }
            }
            core::hint::spin_loop();
        }
        None
    }

    fn send_command(&mut self, param: u64, control: u32) -> Option<Trb> {
        let _ = self.cmd_ring.push(param, 0, control);
        self.ring_doorbell(0, 0);
        let ev = self.wait_event(TRB_TYPE_CMD_COMPLETION, 500)?;
        let cc = (ev.status >> 24) & 0xFF;
        if cc == 1 { Some(ev) } else { None }
    }

    fn control_transfer(
        &mut self,
        slot_id: u8,
        ep0_ring: &mut TrbRing,
        setup_packet: u64,
        data_buf: Option<(u64, u16, bool)>, // (phys, len, dir_in)
    ) -> bool {
        let trt = match data_buf {
            None => 0u32,
            Some((_, _, true)) => 3u32,  // IN Data Stage
            Some((_, _, false)) => 2u32, // OUT Data Stage
        };
        // Setup Stage TRB (IDT = bit 6)
        ep0_ring.push(
            setup_packet,
            8,
            (TRB_TYPE_SETUP << 10) | (1 << 6) | (trt << 16),
        );
        if let Some((buf, len, dir_in)) = data_buf {
            let dir = if dir_in { 1u32 << 16 } else { 0 };
            ep0_ring.push(buf, len as u32, (TRB_TYPE_DATA << 10) | dir);
        }
        // Status Stage TRB (IOC = bit 5; direction opposite of data, or IN when no data)
        let status_dir = match data_buf {
            Some((_, _, true)) => 0u32,
            _ => 1u32 << 16,
        };
        ep0_ring.push(0, 0, (TRB_TYPE_STATUS << 10) | (1 << 5) | status_dir);
        self.ring_doorbell(slot_id, 1); // EP0 DCI = 1

        if let Some(ev) = self.wait_event(TRB_TYPE_TRANSFER_EVENT, 500) {
            let cc = (ev.status >> 24) & 0xFF;
            // 1 = Success, 13 = Short Packet (acceptable for descriptor reads)
            return cc == 1 || cc == 13;
        }
        false
    }

    fn ctx_ptr(&self, base: u64, index: usize) -> *mut u32 {
        (base + (index * self.ctx_size) as u64) as *mut u32
    }

    fn probe_port(&mut self, port: u8) {
        let paddr = self.portsc_addr(port);
        let sc = self.rd32(paddr);
        if (sc & 1) == 0 {
            return; // Not connected
        }

        // Preserve PP (bit 9), clear RW1C status bits, assert PR (bit 4) for USB2 ports
        // (USB3 ports transition to U0 automatically; reset only if not already PED=1).
        if (sc & (1 << 1)) == 0 {
            let reset_val = (sc & 0x0E01_C3E0) | (1 << 4);
            self.wr32(paddr, reset_val);
            let deadline = cpu::now_nanos() + 150_000_000;
            while cpu::now_nanos() < deadline {
                let cur = self.rd32(paddr);
                if (cur & (1 << 4)) == 0 && (cur & (1 << 1)) != 0 {
                    break;
                }
                cpu::busy_wait_us(1000);
            }
        }
        let sc = self.rd32(paddr);
        if (sc & (1 << 1)) == 0 {
            return; // Port not enabled
        }
        // Clear any change bits (RW1C bits 17..23).
        self.wr32(paddr, (sc & 0x0E01_C3E0) | (sc & 0x00FE_0000));
        cpu::busy_wait_us(10_000);

        let speed = (sc >> 10) & 0xF;
        let ep0_mps: u32 = match speed {
            1 | 2 => 8,   // Full / Low speed
            3 => 64,      // High speed
            4..=6 => 512, // SuperSpeed
            _ => 64,
        };

        // 1. Enable Slot
        let Some(ev) = self.send_command(0, TRB_TYPE_ENABLE_SLOT << 10) else {
            return;
        };
        let slot_id = ((ev.control >> 24) & 0xFF) as u8;
        if slot_id == 0 {
            return;
        }

        // 2. Allocate Device Context, Input Context, EP0 Transfer Ring
        let Some(dev_ctx) = mem::alloc_dma_zeroed(32 * self.ctx_size, 4096) else {
            return;
        };
        let Some(in_ctx) = mem::alloc_dma_zeroed(33 * self.ctx_size, 4096) else {
            return;
        };
        let Some(mut ep0_ring) = TrbRing::new() else {
            return;
        };
        unsafe {
            ((self.dcbaa + (slot_id as u64) * 8) as *mut u64).write_volatile(dev_ctx);
        }

        // Input Control Context (index 0): Add Slot (bit 0) + EP0 (bit 1)
        unsafe {
            let icc = self.ctx_ptr(in_ctx, 0);
            icc.add(1).write_volatile(0x3);
            // Slot Context (index 1): Context Entries = 1, Speed, Root Hub Port Number
            let slot_c = self.ctx_ptr(in_ctx, 1);
            slot_c.add(0).write_volatile((1 << 27) | (speed << 20));
            slot_c.add(1).write_volatile((port as u32) << 16);
            // EP0 Context (index 2): CErr=3, EPType=4 (Control), MPS, TR Dequeue | DCS=1
            let ep0_c = self.ctx_ptr(in_ctx, 2);
            ep0_c
                .add(1)
                .write_volatile((3 << 1) | (4 << 3) | (ep0_mps << 16));
            let deq = ep0_ring.phys | 1;
            ep0_c.add(2).write_volatile(deq as u32);
            ep0_c.add(3).write_volatile((deq >> 32) as u32);
            ep0_c.add(4).write_volatile(8); // Avg TRB len
        }

        // 3. Address Device
        if self
            .send_command(
                in_ctx,
                (TRB_TYPE_ADDRESS_DEVICE << 10) | ((slot_id as u32) << 24),
            )
            .is_none()
        {
            return;
        }
        cpu::busy_wait_us(5_000);

        // 4. Read Configuration Descriptor (first 9 bytes, then full config up to 512 bytes)
        let Some(desc_buf) = mem::alloc_dma_zeroed(4096, 4096) else {
            return;
        };
        let make_setup = |bm_req: u8, b_req: u8, w_val: u16, w_idx: u16, w_len: u16| -> u64 {
            (bm_req as u64)
                | ((b_req as u64) << 8)
                | ((w_val as u64) << 16)
                | ((w_idx as u64) << 32)
                | ((w_len as u64) << 48)
        };

        // GET_DESCRIPTOR(Configuration=2, index=0), 9 bytes
        if !self.control_transfer(slot_id, &mut ep0_ring, make_setup(0x80, 6, 0x0200, 0, 9), Some((desc_buf, 9, true))) {
            return;
        }
        let cfg_hdr = unsafe { core::slice::from_raw_parts(desc_buf as *const u8, 9) };
        let total_len = u16::from_le_bytes([cfg_hdr[2], cfg_hdr[3]]).min(1024);
        let cfg_val = cfg_hdr[5];
        if total_len < 9 {
            return;
        }
        if !self.control_transfer(
            slot_id,
            &mut ep0_ring,
            make_setup(0x80, 6, 0x0200, 0, total_len),
            Some((desc_buf, total_len, true)),
        ) {
            return;
        }

        let cfg = unsafe { core::slice::from_raw_parts(desc_buf as *const u8, total_len as usize) };
        struct FoundHid {
            kind: HidKind,
            iface_num: u8,
            ep_addr: u8,
            mps: u16,
            interval: u8,
        }
        let mut found: Vec<FoundHid> = Vec::new();
        let mut cur_hid: Option<(HidKind, u8)> = None;
        let mut pos = 0usize;
        while pos + 2 <= cfg.len() {
            let len = cfg[pos] as usize;
            let dtype = cfg[pos + 1];
            if len < 2 || pos + len > cfg.len() {
                break;
            }
            if dtype == 4 && len >= 9 {
                // Interface Descriptor
                let iface_num = cfg[pos + 2];
                let cls = cfg[pos + 5];
                let sub = cfg[pos + 6];
                let proto = cfg[pos + 7];
                if cls == 3 && sub == 1 && proto == 1 {
                    cur_hid = Some((HidKind::Keyboard, iface_num));
                } else if cls == 3 && sub == 1 && proto == 2 {
                    cur_hid = Some((HidKind::Mouse, iface_num));
                } else {
                    cur_hid = None;
                }
            } else if dtype == 5 && len >= 7 {
                // Endpoint Descriptor
                if let Some((kind, iface_num)) = cur_hid {
                    let ep_addr = cfg[pos + 2];
                    let attrs = cfg[pos + 3];
                    let mps = u16::from_le_bytes([cfg[pos + 4], cfg[pos + 5]]) & 0x7FF;
                    let interval = cfg[pos + 6];
                    // Interrupt IN endpoint
                    if (ep_addr & 0x80) != 0 && (attrs & 0x3) == 3 {
                        found.push(FoundHid {
                            kind,
                            iface_num,
                            ep_addr,
                            mps: mps.max(8),
                            interval,
                        });
                        cur_hid = None;
                    }
                }
            }
            pos += len;
        }

        if found.is_empty() {
            return;
        }

        // 5. SET_CONFIGURATION
        if !self.control_transfer(slot_id, &mut ep0_ring, make_setup(0x00, 9, cfg_val as u16, 0, 0), None) {
            return;
        }

        // 6. Configure each HID Interrupt IN endpoint and switch interface to Boot Protocol.
        for hid in found {
            // SET_PROTOCOL(Boot = 0) and SET_IDLE(0)
            let _ = self.control_transfer(
                slot_id,
                &mut ep0_ring,
                make_setup(0x21, 0x0B, 0, hid.iface_num as u16, 0),
                None,
            );
            let _ = self.control_transfer(
                slot_id,
                &mut ep0_ring,
                make_setup(0x21, 0x0A, 0, hid.iface_num as u16, 0),
                None,
            );

            let ep_num = hid.ep_addr & 0x0F;
            let dci = ep_num * 2 + 1; // IN endpoint DCI
            if dci < 2 || dci >= 32 {
                continue;
            }
            let Some(mut ep_ring) = TrbRing::new() else {
                continue;
            };
            let Some(report_buf) = mem::alloc_dma_zeroed(4096, 4096) else {
                continue;
            };

            // Compute xHCI Interval field based on port speed.
            let xhci_interval: u32 = match speed {
                1 | 2 => {
                    // FS/LS: bInterval is in ms; xHCI encodes 2^(val-3) ms -> val = ceil(log2(ms)) + 3
                    let ms = (hid.interval as u32).clamp(1, 255);
                    let mut p = 0u32;
                    while (1u32 << p) < ms && p < 8 {
                        p += 1;
                    }
                    (p + 3).clamp(3, 10)
                }
                _ => (hid.interval as u32).saturating_sub(1).clamp(0, 15),
            };

            // Zero input context and populate Slot Context + Endpoint Context.
            unsafe {
                core::ptr::write_bytes(in_ctx as *mut u8, 0, 33 * self.ctx_size);
                // Copy current output Slot Context into Input Slot Context (index 1).
                core::ptr::copy_nonoverlapping(
                    dev_ctx as *const u8,
                    (in_ctx + self.ctx_size as u64) as *mut u8,
                    self.ctx_size,
                );
                let icc = self.ctx_ptr(in_ctx, 0);
                icc.add(1).write_volatile(1 | (1u32 << dci)); // Add Slot + DCI

                let slot_c = self.ctx_ptr(in_ctx, 1);
                let cur_d0 = slot_c.add(0).read_volatile();
                let cur_entries = (cur_d0 >> 27) & 0x1F;
                let new_entries = cur_entries.max(dci as u32);
                slot_c
                    .add(0)
                    .write_volatile((cur_d0 & !(0x1F << 27)) | (new_entries << 27));

                // Endpoint context at index (dci + 1)
                let ep_c = self.ctx_ptr(in_ctx, dci as usize + 1);
                ep_c.add(0).write_volatile(xhci_interval << 16);
                // CErr=3, EPType=7 (Interrupt IN), MaxPacketSize
                ep_c.add(1)
                    .write_volatile((3 << 1) | (7 << 3) | ((hid.mps as u32) << 16));
                let deq = ep_ring.phys | 1;
                ep_c.add(2).write_volatile(deq as u32);
                ep_c.add(3).write_volatile((deq >> 32) as u32);
                // Avg TRB len + Max ESIT Payload Lo
                ep_c.add(4)
                    .write_volatile((hid.mps as u32) | ((hid.mps as u32) << 16));
            }

            if self
                .send_command(
                    in_ctx,
                    (TRB_TYPE_CONFIGURE_EP << 10) | ((slot_id as u32) << 24),
                )
                .is_none()
            {
                continue;
            }

            let report_len = match hid.kind {
                HidKind::Keyboard => 8u16.min(hid.mps),
                HidKind::Mouse => 4u16.min(hid.mps),
            };
            // Queue initial Interrupt IN Normal TRB (IOC | ISP).
            ep_ring.push(
                report_buf,
                report_len as u32,
                (TRB_TYPE_NORMAL << 10) | (1 << 5) | (1 << 2),
            );
            self.ring_doorbell(slot_id, dci as u32);

            let label = match hid.kind {
                HidKind::Keyboard => "USB HID Keyboard",
                HidKind::Mouse => "USB HID Mouse",
            };
            kprintln!(
                "[xhci] Bound {} on port {} (slot={} dci={} speed={} mps={})",
                label,
                port,
                slot_id,
                dci,
                speed,
                hid.mps
            );

            self.hid_devices.push(HidDevice {
                slot_id,
                dci,
                kind: hid.kind,
                report_len,
                ring: ep_ring,
                buf_phys: report_buf,
                prev_keys: [0; 6],
                prev_mouse_btns: 0,
            });
        }
    }

    fn poll(&mut self) {
        for _ in 0..64 {
            let Some(ev) = self.pop_event() else {
                break;
            };
            let ty = (ev.control >> 10) & 0x3F;
            if ty != TRB_TYPE_TRANSFER_EVENT {
                continue;
            }
            let slot_id = ((ev.control >> 24) & 0xFF) as u8;
            let dci = ((ev.control >> 16) & 0x1F) as u8;
            let cc = (ev.status >> 24) & 0xFF;

            let mut requeue: Option<(u8, u8)> = None;
            for dev in &mut self.hid_devices {
                if dev.slot_id == slot_id && dev.dci == dci {
                    if cc == 1 || cc == 13 {
                        let data = unsafe {
                            core::slice::from_raw_parts(
                                dev.buf_phys as *const u8,
                                dev.report_len as usize,
                            )
                        };
                        match dev.kind {
                            HidKind::Keyboard if data.len() >= 8 => {
                                let mods = data[0];
                                let shift = (mods & 0x22) != 0;
                                let ctrl = (mods & 0x11) != 0;
                                let alt = (mods & 0x44) != 0;
                                let keys = &data[2..8];

                                // Emit KeyUp for keys no longer held
                                for &old in &dev.prev_keys {
                                    if old >= 4 && !keys.contains(&old) {
                                        if let Some(kc) = input::hid_usage_to_keycode(old, shift) {
                                            input::push(InputEvent::KeyUp { key: kc });
                                        }
                                    }
                                }
                                // Emit KeyDown for newly pressed keys
                                for &k in keys {
                                    if k >= 4 && !dev.prev_keys.contains(&k) {
                                        if let Some(kc) = input::hid_usage_to_keycode(k, shift) {
                                            input::push(InputEvent::KeyDown {
                                                key: kc,
                                                shift,
                                                ctrl,
                                                alt,
                                            });
                                        }
                                    }
                                }
                                dev.prev_keys.copy_from_slice(keys);
                            }
                            HidKind::Mouse if data.len() >= 3 => {
                                let btns = data[0] & 0x07;
                                let dx = data[1] as i8 as i32;
                                let dy = data[2] as i8 as i32;
                                let wheel = if data.len() >= 4 { data[3] as i8 } else { 0 };
                                if dx != 0 || dy != 0 {
                                    input::push(InputEvent::MouseMove { dx, dy });
                                }
                                let changed = btns ^ dev.prev_mouse_btns;
                                for bit in 0..3u8 {
                                    if (changed & (1 << bit)) != 0 {
                                        input::push(InputEvent::MouseButton {
                                            button: bit,
                                            pressed: (btns & (1 << bit)) != 0,
                                        });
                                    }
                                }
                                dev.prev_mouse_btns = btns;
                                if wheel != 0 {
                                    input::push(InputEvent::MouseWheel { delta: wheel });
                                }
                            }
                            _ => {}
                        }
                    }

                    // Re-queue next Interrupt IN TRB.
                    dev.ring.push(
                        dev.buf_phys,
                        dev.report_len as u32,
                        (TRB_TYPE_NORMAL << 10) | (1 << 5) | (1 << 2),
                    );
                    requeue = Some((dev.slot_id, dev.dci));
                    break;
                }
            }
            if let Some((s, d)) = requeue {
                self.ring_doorbell(s, d as u32);
            }
        }
    }
}

static mut XHCI: Option<XhciController> = None;

pub fn init(pml4_phys: u64) {
    let Some(pci) = find_xhci_pci() else {
        kprintln!("[xhci] No xHCI controller found on PCI");
        return;
    };
    kprintln!(
        "[xhci] PCI {:02x}:{:02x}.{} [{:04x}:{:04x}] BAR0={:#x}",
        pci.bus,
        pci.dev,
        pci.func,
        pci.vendor,
        pci.device,
        pci.bar0
    );

    // Map BAR0 as Uncacheable (PAT UC) before touching MMIO registers.
    cpu::set_identity_range_cache_attr(pml4_phys, pci.bar0, 0x1_0000, cpu::PAT_UC_BITS);

    let cap_base = pci.bar0;
    let cap_len = unsafe { (cap_base as *const u8).read_volatile() } as u64;
    let hcsparams1 = unsafe { ((cap_base + 0x04) as *const u32).read_volatile() };
    let hcsparams2 = unsafe { ((cap_base + 0x08) as *const u32).read_volatile() };
    let hccparams1 = unsafe { ((cap_base + 0x10) as *const u32).read_volatile() };
    let dboff = unsafe { ((cap_base + 0x14) as *const u32).read_volatile() } & !0x3;
    let rtsoff = unsafe { ((cap_base + 0x18) as *const u32).read_volatile() } & !0x1F;

    let max_slots = (hcsparams1 & 0xFF).min(64) as u8;
    let max_ports = ((hcsparams1 >> 24) & 0xFF) as u8;
    let max_scratch_hi = (hcsparams2 >> 21) & 0x1F;
    let max_scratch_lo = (hcsparams2 >> 27) & 0x1F;
    let max_scratchpads = ((max_scratch_hi << 5) | max_scratch_lo) as usize;
    let ctx_size = if (hccparams1 & (1 << 2)) != 0 { 64 } else { 32 };
    let xecp = (hccparams1 >> 16) & 0xFFFF;

    let op_base = cap_base + cap_len;
    let db_base = cap_base + dboff as u64;
    let rt_base = cap_base + rtsoff as u64;

    let Some(dcbaa) = mem::alloc_dma_zeroed(4096, 4096) else {
        return;
    };
    let Some(cmd_ring) = TrbRing::new() else {
        return;
    };
    let Some(evt_ring_phys) = mem::alloc_dma_zeroed(EVENT_RING_LEN * 16, 4096) else {
        return;
    };
    let Some(erst_phys) = mem::alloc_dma_zeroed(4096, 4096) else {
        return;
    };

    let mut ctrl = XhciController {
        cap_base,
        op_base,
        db_base,
        rt_base,
        ctx_size,
        max_ports,
        dcbaa,
        cmd_ring,
        evt_ring_phys,
        evt_deq: 0,
        evt_cycle: true,
        hid_devices: Vec::new(),
    };

    ctrl.bios_handoff(xecp);

    // Halt controller
    let cmd = ctrl.rd32(op_base);
    ctrl.wr32(op_base, cmd & !1);
    for _ in 0..100 {
        if (ctrl.rd32(op_base + 0x04) & 1) != 0 {
            break;
        }
        cpu::busy_wait_us(1000);
    }

    // Reset controller (HCRST)
    ctrl.wr32(op_base, 1 << 1);
    for _ in 0..500 {
        let c = ctrl.rd32(op_base);
        let s = ctrl.rd32(op_base + 0x04);
        if (c & (1 << 1)) == 0 && (s & (1 << 11)) == 0 {
            break;
        }
        cpu::busy_wait_us(1000);
    }

    // MaxSlotsEn
    ctrl.wr32(op_base + 0x38, max_slots as u32);

    // Scratchpad buffers (required on Intel 600-series xHCI!)
    if max_scratchpads > 0 {
        if let Some(sp_arr) = mem::alloc_dma_zeroed((max_scratchpads * 8).max(4096), 4096) {
            for i in 0..max_scratchpads {
                if let Some(page) = mem::alloc_zeroed_page_phys() {
                    unsafe {
                        ((sp_arr + (i as u64) * 8) as *mut u64).write_volatile(page);
                    }
                }
            }
            unsafe {
                (dcbaa as *mut u64).write_volatile(sp_arr);
            }
        }
    }

    // DCBAAP + CRCR
    ctrl.wr64(op_base + 0x30, dcbaa);
    ctrl.wr64(op_base + 0x18, ctrl.cmd_ring.phys | 1);

    // Event Ring Segment Table + Interrupter 0
    unsafe {
        (erst_phys as *mut ErstEntry).write_volatile(ErstEntry {
            base: evt_ring_phys,
            size: EVENT_RING_LEN as u32,
            reserved: 0,
        });
    }
    let ir0 = rt_base + 0x20;
    ctrl.wr32(ir0 + 0x08, 1); // ERSTSZ = 1
    ctrl.wr64(ir0 + 0x18, evt_ring_phys); // ERDP
    ctrl.wr64(ir0 + 0x10, erst_phys); // ERSTBA
    ctrl.wr32(ir0 + 0x00, 0); // IMAN polling mode

    // Run controller (RS = 1)
    ctrl.wr32(op_base, 1);
    for _ in 0..200 {
        if (ctrl.rd32(op_base + 0x04) & 1) == 0 {
            break;
        }
        cpu::busy_wait_us(1000);
    }

    kprintln!(
        "[xhci] Running (ports={}, slots={}, ctx={}B, scratch={})",
        max_ports,
        max_slots,
        ctx_size,
        max_scratchpads
    );

    // Give connected root hub devices a short settle window, then probe all ports.
    cpu::busy_wait_us(30_000);
    for port in 1..=max_ports {
        ctrl.probe_port(port);
    }
    kprintln!("[xhci] Bound {} HID device(s)", ctrl.hid_devices.len());

    unsafe {
        XHCI = Some(ctrl);
    }
}

pub fn poll() {
    unsafe {
        if let Some(ctrl) = (*(&raw mut XHCI)).as_mut() {
            ctrl.poll();
        }
    }
}
