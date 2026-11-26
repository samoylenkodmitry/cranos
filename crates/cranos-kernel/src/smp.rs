use alloc::vec::Vec;
use core::arch::{asm, global_asm};
use core::arch::x86_64::__cpuid_count;
use core::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use crate::acpi;
use crate::cpu;
use crate::gdt;
use crate::idt;
use crate::kprintln;
use crate::mem;

const IA32_APIC_BASE_MSR: u32 = 0x1B;
const IA32_X2APIC_ICR_MSR: u32 = 0x830;
const IA32_X2APIC_ID_MSR: u32 = 0x802;
const IA32_X2APIC_SIVR_MSR: u32 = 0x80F;

const LAPIC_ID: usize = 0x020;
const LAPIC_SVR: usize = 0x0F0;
const LAPIC_ICR_LO: usize = 0x300;
const LAPIC_ICR_HI: usize = 0x310;

// Fixed physical addresses in conventional low memory (< 1 MiB) for real-mode SIPI startup.
const TRAMPOLINE_PHYS: u64 = 0x8000; // SIPI vector 0x08
const MAILBOX_PHYS: u64 = 0x7000;

const MAX_WORKERS: usize = 64;

#[repr(C)]
struct ApMailbox {
    pml4_phys: u64,   // +0
    entry_addr: u64,  // +8
    stack_top: u64,   // +16
    worker_id: u32,   // +24
    ready: AtomicU32, // +28
}

// 16-bit real mode -> 32-bit protected mode -> 64-bit long mode AP trampoline.
// Copied verbatim to physical address 0x8000 before sending INIT-SIPI-SIPI.
global_asm!(
    ".section .rodata",
    ".global __ap_trampoline_start",
    ".global __ap_trampoline_end",
    "__ap_trampoline_start:",
    ".code16",
    "cli",
    "cld",
    "xor ax, ax",
    "mov ds, ax",
    "mov es, ax",
    "mov ss, ax",
    // Load temporary 32/64-bit GDT located at 0x8000 + (__ap_gdt_desc - __ap_trampoline_start)
    // Encoding of 16-bit `lgdt [disp16]` with DS=0: 0F 01 16 <disp16>
    ".byte 0x0F, 0x01, 0x16",
    ".word 0x8000 + (__ap_gdt_desc - __ap_trampoline_start)",
    // Enter 32-bit protected mode (CR0.PE = 1)
    "mov eax, cr0",
    "or eax, 1",
    "mov cr0, eax",
    // Far jump to 32-bit code selector 0x08
    ".byte 0x66, 0xEA",
    ".long 0x8000 + (__ap_pm32 - __ap_trampoline_start)",
    ".word 0x0008",

    ".code32",
    "__ap_pm32:",
    "mov ax, 0x10",
    "mov ds, ax",
    "mov es, ax",
    "mov ss, ax",
    "mov fs, ax",
    "mov gs, ax",
    // Enable PAE (CR4.PAE = 1 << 5)
    "mov eax, cr4",
    "or eax, 0x20",
    "mov cr4, eax",
    // Load PML4 physical address from mailbox at 0x7000
    "mov eax, dword ptr [0x7000]",
    "mov cr3, eax",
    // Enable Long Mode (IA32_EFER.LME = 1 << 8)
    "mov ecx, 0xC0000080",
    "rdmsr",
    "or eax, 0x100",
    "wrmsr",
    // Enable Paging (CR0.PG = 1 << 31)
    "mov eax, cr0",
    "or eax, 0x80000000",
    "mov cr0, eax",
    // Far jump to 64-bit code selector 0x18
    ".byte 0xEA",
    ".long 0x8000 + (__ap_lm64 - __ap_trampoline_start)",
    ".word 0x0018",

    ".code64",
    "__ap_lm64:",
    "mov ax, 0x10",
    "mov ds, ax",
    "mov es, ax",
    "mov ss, ax",
    // Load per-AP stack and worker_id from mailbox at 0x7000
    "mov rsp, qword ptr [0x7010]",
    "sub rsp, 8",
    "xor rbp, rbp",
    "mov edi, dword ptr [0x7018]",
    "mov rax, qword ptr [0x7008]",
    "jmp rax",

    ".balign 8",
    "__ap_gdt:",
    ".quad 0x0000000000000000", // 0x00 null
    ".quad 0x00CF9A000000FFFF", // 0x08 32-bit code
    ".quad 0x00CF92000000FFFF", // 0x10 data
    ".quad 0x00AF9A000000FFFF", // 0x18 64-bit code
    "__ap_gdt_desc:",
    ".word 31",
    ".long 0x8000 + (__ap_gdt - __ap_trampoline_start)",
    "__ap_trampoline_end:",
);

unsafe extern "C" {
    static __ap_trampoline_start: u8;
    static __ap_trampoline_end: u8;
}

type WorkFn = fn(worker_id: usize, total_workers: usize, ctx: *const ());

#[repr(align(64))]
struct WorkDispatch {
    epoch: AtomicU64,
    active_workers: AtomicUsize,
    fn_ptr: AtomicUsize,
    ctx_ptr: AtomicPtr<()>,
}

#[repr(align(64))]
struct DoneCounter {
    count: AtomicU32,
}

static DISPATCH: WorkDispatch = WorkDispatch {
    epoch: AtomicU64::new(0),
    active_workers: AtomicUsize::new(1),
    fn_ptr: AtomicUsize::new(0),
    ctx_ptr: AtomicPtr::new(core::ptr::null_mut()),
};

static DONE: DoneCounter = DoneCounter {
    count: AtomicU32::new(0),
};

static ONLINE_CORES: AtomicUsize = AtomicUsize::new(1);
static PHYSICAL_CORES: AtomicUsize = AtomicUsize::new(1);

// Tracks whether each worker_id is a primary physical core (SMT thread 0).
static WORKER_IS_PRIMARY: [AtomicU32; MAX_WORKERS] = [const { AtomicU32::new(0) }; MAX_WORKERS];

fn is_smt_secondary_thread() -> bool {
    // 1. CPUID leaf 0xB subleaf 0: SMT level width in EAX[4:0], x2APIC ID in EDX
    //    (supported on Intel Core and AMD Zen 2 / Ryzen 3950X).
    let max_leaf = core::arch::x86_64::__cpuid(0).eax;
    if max_leaf >= 0xB {
        let smt = __cpuid_count(0xB, 0);
        let level_type = (smt.ecx >> 8) & 0xFF;
        let shift = smt.eax & 0x1F;
        if level_type == 1 && shift > 0 {
            let smt_mask = (1u32 << shift) - 1;
            return (smt.edx & smt_mask) != 0;
        }
    }
    // 2. Fallback: AMD Topology Extensions leaf 0x8000_001E (EBX[15:8] = threads_per_core - 1).
    let max_ext = core::arch::x86_64::__cpuid(0x8000_0000).eax;
    if max_ext >= 0x8000_001E {
        let ext1 = core::arch::x86_64::__cpuid(0x8000_0001);
        if (ext1.ecx & (1 << 22)) != 0 {
            let topo = core::arch::x86_64::__cpuid(0x8000_001E);
            let threads_per_core = ((topo.ebx >> 8) & 0xFF) + 1;
            if threads_per_core > 1 {
                return (topo.eax % threads_per_core) != 0;
            }
        }
    }
    false
}

extern "C" fn ap_entry(worker_id: u32) -> ! {
    gdt::load_ap();
    idt::load();
    cpu::enable_simd_on_current_core();
    cpu::configure_pat_wc();

    let secondary = is_smt_secondary_thread();
    let idx = worker_id as usize;
    if idx < MAX_WORKERS {
        WORKER_IS_PRIMARY[idx].store(u32::from(!secondary), Ordering::Relaxed);
    }
    ONLINE_CORES.fetch_add(1, Ordering::Release);

    let mailbox = unsafe { &*(MAILBOX_PHYS as *const ApMailbox) };

    // Park SMT secondary threads in HLT immediately: memory-bandwidth-bound LLM
    // matmul runs faster on dedicated physical cores, and halting sibling
    // threads avoids unnecessary heat and L1/L2 contention (e.g. Ryzen 9 3950X).
    if secondary {
        mailbox.ready.store(1, Ordering::Release);
        loop {
            unsafe { asm!("cli; hlt", options(nomem, nostack)) };
        }
    }

    // Claim a dense physical-core slot (1..PHYSICAL_CORES) before signalling ready.
    let primary_slot = PHYSICAL_CORES.fetch_add(1, Ordering::AcqRel);
    mailbox.ready.store(1, Ordering::Release);

    // Tight zero-jitter spin-barrier loop during active inference, with gentle
    // PAUSE backoff when idle between prompts to keep laptop thermals cool.
    let mut last_epoch = DISPATCH.epoch.load(Ordering::Acquire);
    let mut idle_spins: u32 = 0;
    loop {
        let cur = DISPATCH.epoch.load(Ordering::Acquire);
        if cur == last_epoch {
            if idle_spins < 200_000 {
                idle_spins += 1;
                core::hint::spin_loop();
            } else {
                for _ in 0..64 {
                    core::hint::spin_loop();
                }
            }
            continue;
        }
        idle_spins = 0;
        last_epoch = cur;
        let total = DISPATCH.active_workers.load(Ordering::Relaxed);
        if primary_slot < total {
            let f_raw = DISPATCH.fn_ptr.load(Ordering::Relaxed);
            if f_raw != 0 {
                let f: WorkFn = unsafe { core::mem::transmute(f_raw) };
                let ctx = DISPATCH.ctx_ptr.load(Ordering::Relaxed) as *const ();
                f(primary_slot, total, ctx);
            }
            DONE.count.fetch_add(1, Ordering::Release);
        }
    }
}

struct Lapic {
    mmio: u64,
    x2apic: bool,
}

impl Lapic {
    fn new(mmio_phys: u64, pml4_phys: u64) -> Self {
        let base_msr = unsafe { cpu::rdmsr(IA32_APIC_BASE_MSR) };
        let x2apic = (base_msr & (1 << 10)) != 0;
        let mmio = if mmio_phys != 0 {
            mmio_phys
        } else {
            base_msr & 0xFFFF_F000
        };
        if !x2apic {
            cpu::set_identity_range_cache_attr(pml4_phys, mmio, 4096, cpu::PAT_UC_BITS);
        }
        Self { mmio, x2apic }
    }

    fn enable(&self) {
        unsafe {
            if self.x2apic {
                let svr = cpu::rdmsr(IA32_X2APIC_SIVR_MSR);
                cpu::wrmsr(IA32_X2APIC_SIVR_MSR, svr | 0x1FF);
            } else {
                let ptr = (self.mmio as usize + LAPIC_SVR) as *mut u32;
                let v = ptr.read_volatile();
                ptr.write_volatile(v | 0x1FF);
            }
        }
    }

    fn current_id(&self) -> u32 {
        unsafe {
            if self.x2apic {
                cpu::rdmsr(IA32_X2APIC_ID_MSR) as u32
            } else {
                let ptr = (self.mmio as usize + LAPIC_ID) as *const u32;
                ptr.read_volatile() >> 24
            }
        }
    }

    fn send_ipi(&self, dest_apic_id: u32, icr_lo: u32) {
        unsafe {
            if self.x2apic {
                let val = ((dest_apic_id as u64) << 32) | (icr_lo as u64);
                cpu::wrmsr(IA32_X2APIC_ICR_MSR, val);
            } else {
                let hi = (self.mmio as usize + LAPIC_ICR_HI) as *mut u32;
                let lo = (self.mmio as usize + LAPIC_ICR_LO) as *mut u32;
                while (lo.read_volatile() & (1 << 12)) != 0 {
                    core::hint::spin_loop();
                }
                hi.write_volatile(dest_apic_id << 24);
                lo.write_volatile(icr_lo);
                while (lo.read_volatile() & (1 << 12)) != 0 {
                    core::hint::spin_loop();
                }
            }
        }
    }
}

pub fn init(rsdp_addr: u64, pml4_phys: u64) {
    WORKER_IS_PRIMARY[0].store(1, Ordering::Relaxed);

    let Some(madt) = acpi::parse_madt(rsdp_addr) else {
        kprintln!("[smp] ACPI MADT not found; running single-core");
        return;
    };

    let lapic = Lapic::new(madt.lapic_addr, pml4_phys);
    lapic.enable();
    let bsp_id = lapic.current_id();

    let mut target_aps: Vec<u32> = madt
        .apic_ids
        .into_iter()
        .filter(|&id| id != bsp_id)
        .collect();
    // Cap workers to MAX_WORKERS - 1.
    if target_aps.len() >= MAX_WORKERS {
        target_aps.truncate(MAX_WORKERS - 1);
    }

    if target_aps.is_empty() {
        kprintln!("[smp] 1 core detected (BSP apic_id={})", bsp_id);
        return;
    }

    // Copy 16-bit -> 64-bit trampoline to 0x8000.
    unsafe {
        let start = &__ap_trampoline_start as *const u8;
        let end = &__ap_trampoline_end as *const u8;
        let len = end as usize - start as usize;
        core::ptr::copy_nonoverlapping(start, TRAMPOLINE_PHYS as *mut u8, len);
    }

    let mut next_worker_id = 1u32;
    for apic_id in target_aps {
        const AP_STACK_SIZE: usize = 256 * 1024;
        let Some(stack_phys) = mem::alloc_dma_zeroed(AP_STACK_SIZE, 4096) else {
            break;
        };

        let mailbox = unsafe { &mut *(MAILBOX_PHYS as *mut ApMailbox) };
        mailbox.pml4_phys = pml4_phys;
        mailbox.entry_addr = ap_entry as *const () as u64;
        mailbox.stack_top = stack_phys + AP_STACK_SIZE as u64;
        mailbox.worker_id = next_worker_id;
        mailbox.ready.store(0, Ordering::Release);

        // INIT IPI (assert, level)
        lapic.send_ipi(apic_id, 0x0000_4500);
        cpu::busy_wait_us(10_000);
        // SIPI #1 (vector 0x08 -> 0x8000)
        lapic.send_ipi(apic_id, 0x0000_4608);

        let mut booted = false;
        for attempt in 0..2 {
            let deadline = cpu::now_nanos() + 20_000_000; // 20 ms
            while cpu::now_nanos() < deadline {
                if mailbox.ready.load(Ordering::Acquire) != 0 {
                    booted = true;
                    break;
                }
                core::hint::spin_loop();
            }
            if booted {
                break;
            }
            if attempt == 0 {
                // SIPI #2 retry per Intel MP spec
                lapic.send_ipi(apic_id, 0x0000_4608);
            }
        }

        if booted {
            next_worker_id += 1;
        }
    }

    kprintln!(
        "[smp] Online: {} logical cores ({} physical cores), BSP apic_id={}, x2apic={}",
        online_cores(),
        physical_cores(),
        bsp_id,
        lapic.x2apic
    );
}

pub fn online_cores() -> usize {
    ONLINE_CORES.load(Ordering::Acquire)
}

pub fn physical_cores() -> usize {
    PHYSICAL_CORES.load(Ordering::Acquire).max(1)
}

/// Runs `f(worker_id, num_workers)` across `num_workers` bare-metal cores
/// simultaneously using our lock-free atomic spin barrier.
/// Worker 0 runs on the calling core (BSP); workers 1..num_workers run on APs.
pub fn parallel_run<F: Fn(usize, usize) + Sync>(max_workers: usize, f: &F) {
    let workers = max_workers.min(physical_cores()).max(1);
    if workers == 1 {
        f(0, 1);
        return;
    }

    fn trampoline<F: Fn(usize, usize) + Sync>(worker_id: usize, total: usize, ctx: *const ()) {
        let func = unsafe { &*(ctx as *const F) };
        func(worker_id, total);
    }

    DONE.count.store(0, Ordering::Relaxed);
    DISPATCH.active_workers.store(workers, Ordering::Relaxed);
    DISPATCH
        .fn_ptr
        .store(trampoline::<F> as *const () as usize, Ordering::Relaxed);
    DISPATCH
        .ctx_ptr
        .store(f as *const F as *mut (), Ordering::Relaxed);

    // Release fence via epoch increment wakes all APs within tens of nanoseconds.
    DISPATCH.epoch.fetch_add(1, Ordering::Release);

    // Execute worker 0 on BSP immediately.
    f(0, workers);

    let expected = (workers - 1) as u32;
    while DONE.count.load(Ordering::Acquire) < expected {
        core::hint::spin_loop();
    }
}

/// Keeps `asm` import live and provides a light memory pause helper.
#[inline]
pub fn cpu_relax() {
    unsafe { asm!("pause", options(nomem, nostack, preserves_flags)) };
}
