use core::arch::asm;
use core::arch::x86_64::{__cpuid, __cpuid_count, _rdtsc};
use core::sync::atomic::{AtomicU64, Ordering};

use crate::serial::{inb, outb};

const IA32_PAT_MSR: u32 = 0x277;

#[derive(Clone, Copy, Debug)]
pub struct CpuFeatures {
    pub brand: [u8; 48],
    pub brand_len: usize,
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
    pub avx_vnni: bool,
    pub x2apic: bool,
    pub pat: bool,
    pub pdpe1gb: bool,
}

impl CpuFeatures {
    pub fn brand_str(&self) -> &str {
        core::str::from_utf8(&self.brand[..self.brand_len])
            .unwrap_or("x86_64")
            .trim()
    }
}

pub fn detect_features() -> CpuFeatures {
    let max_leaf = __cpuid(0).eax;
    let leaf1 = __cpuid(1);
    let leaf7_0 = if max_leaf >= 7 {
        __cpuid_count(7, 0)
    } else {
        core::arch::x86_64::CpuidResult {
            eax: 0,
            ebx: 0,
            ecx: 0,
            edx: 0,
        }
    };
    let leaf7_1 = if max_leaf >= 7 && leaf7_0.eax >= 1 {
        __cpuid_count(7, 1)
    } else {
        core::arch::x86_64::CpuidResult {
            eax: 0,
            ebx: 0,
            ecx: 0,
            edx: 0,
        }
    };
    let max_ext = __cpuid(0x8000_0000).eax;
    let ext1 = if max_ext >= 0x8000_0001 {
        __cpuid(0x8000_0001)
    } else {
        core::arch::x86_64::CpuidResult {
            eax: 0,
            ebx: 0,
            ecx: 0,
            edx: 0,
        }
    };

    let mut brand = [0u8; 48];
    let mut brand_len = 0usize;
    if max_ext >= 0x8000_0004 {
        for (i, leaf) in [0x8000_0002u32, 0x8000_0003, 0x8000_0004].iter().enumerate() {
            let r = __cpuid(*leaf);
            brand[i * 16..i * 16 + 4].copy_from_slice(&r.eax.to_le_bytes());
            brand[i * 16 + 4..i * 16 + 8].copy_from_slice(&r.ebx.to_le_bytes());
            brand[i * 16 + 8..i * 16 + 12].copy_from_slice(&r.ecx.to_le_bytes());
            brand[i * 16 + 12..i * 16 + 16].copy_from_slice(&r.edx.to_le_bytes());
        }
        brand_len = brand.iter().position(|&b| b == 0).unwrap_or(48);
    }

    let has_osxsave = (leaf1.ecx & (1 << 27)) != 0;
    let xcr0_avx = if has_osxsave {
        let eax: u32;
        unsafe {
            asm!("xgetbv", in("ecx") 0u32, out("eax") eax, out("edx") _, options(nomem, nostack));
        }
        (eax & 0x6) == 0x6
    } else {
        false
    };

    CpuFeatures {
        brand,
        brand_len,
        avx2: xcr0_avx && ((leaf7_0.ebx & (1 << 5)) != 0),
        fma: xcr0_avx && ((leaf1.ecx & (1 << 12)) != 0),
        f16c: xcr0_avx && ((leaf1.ecx & (1 << 29)) != 0),
        avx_vnni: xcr0_avx && ((leaf7_1.eax & (1 << 4)) != 0),
        x2apic: (leaf1.ecx & (1 << 21)) != 0,
        pat: (leaf1.edx & (1 << 16)) != 0,
        pdpe1gb: (ext1.edx & (1 << 26)) != 0,
    }
}

/// Enables SSE/AVX/OSXSAVE on the calling CPU core (used by both BSP and APs).
pub fn enable_simd_on_current_core() {
    unsafe {
        let mut cr0: u64;
        asm!("mov {}, cr0", out(reg) cr0, options(nomem, nostack));
        cr0 &= !((1 << 2) | (1 << 3)); // Clear EM and TS
        cr0 |= 1 << 1; // Set MP
        asm!("mov cr0, {}", in(reg) cr0, options(nomem, nostack));

        let mut cr4: u64;
        asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack));
        cr4 |= (1 << 9) | (1 << 10); // OSFXSR | OSXMMEXCPT
        let leaf1 = __cpuid(1);
        let has_xsave = (leaf1.ecx & (1 << 26)) != 0;
        let has_avx = (leaf1.ecx & (1 << 28)) != 0;
        if has_xsave {
            cr4 |= 1 << 18; // OSXSAVE
        }
        asm!("mov cr4, {}", in(reg) cr4, options(nomem, nostack));

        if has_xsave && has_avx {
            let mut eax: u32;
            let mut edx: u32;
            asm!(
                "xgetbv",
                in("ecx") 0u32,
                out("eax") eax,
                out("edx") edx,
                options(nomem, nostack)
            );
            eax |= 0x7;
            asm!(
                "xsetbv",
                in("ecx") 0u32,
                in("eax") eax,
                in("edx") edx,
                options(nomem, nostack)
            );
        }
    }
}

#[inline]
pub unsafe fn rdmsr(msr: u32) -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        asm!("rdmsr", in("ecx") msr, out("eax") lo, out("edx") hi, options(nomem, nostack));
    }
    ((hi as u64) << 32) | (lo as u64)
}

#[inline]
pub unsafe fn wrmsr(msr: u32, val: u64) {
    let lo = val as u32;
    let hi = (val >> 32) as u32;
    unsafe {
        asm!("wrmsr", in("ecx") msr, in("eax") lo, in("edx") hi, options(nomem, nostack));
    }
}

/// Configures IA32_PAT on the current core so:
///   PAT0 = WB (0x06), PAT1 = WC (0x01), PAT2 = UC- (0x07), PAT3 = UC (0x00)
///   PAT4 = WB (0x06), PAT5 = WC (0x01), PAT6 = UC- (0x07), PAT7 = UC (0x00)
/// This makes PWT=1, PCD=0 select Write-Combining for our linear framebuffer.
pub fn configure_pat_wc() {
    let leaf1 = __cpuid(1);
    if (leaf1.edx & (1 << 16)) == 0 {
        return;
    }
    let pat_val: u64 = 0x0007_0106_0007_0106;
    unsafe {
        wrmsr(IA32_PAT_MSR, pat_val);
    }
}

/// Splits any 1 GiB page containing `phys..phys+size` into 2 MiB pages inside
/// the identity map and applies `cache_bits` (e.g. `1 << 3` = PWT for PAT1=WC,
/// or `(1 << 3) | (1 << 4)` = PWT|PCD for PAT3=UC).
pub fn set_identity_range_cache_attr(pml4_phys: u64, phys: u64, size: u64, cache_bits: u64) {
    if pml4_phys == 0 || size == 0 {
        return;
    }
    const PRESENT: u64 = 1 << 0;
    const WRITABLE: u64 = 1 << 1;
    const HUGE: u64 = 1 << 7;
    const CACHE_MASK: u64 = (1 << 3) | (1 << 4) | (1 << 12);
    const ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;

    let start = phys & !((2 << 20) - 1);
    let end = (phys.saturating_add(size) + (2 << 20) - 1) & !((2 << 20) - 1);

    unsafe {
        let pml4 = &mut *(pml4_phys as *mut [u64; 512]);
        let mut addr = start;
        while addr < end {
            let i4 = ((addr >> 39) & 0x1FF) as usize;
            let i3 = ((addr >> 30) & 0x1FF) as usize;
            let i2 = ((addr >> 21) & 0x1FF) as usize;

            if (pml4[i4] & PRESENT) == 0 {
                let Some(pdpt_phys) = crate::mem::alloc_zeroed_page_phys() else {
                    return;
                };
                pml4[i4] = pdpt_phys | PRESENT | WRITABLE;
            }
            let pdpt = &mut *((pml4[i4] & ADDR_MASK) as *mut [u64; 512]);

            if (pdpt[i3] & PRESENT) == 0 {
                let Some(pd_phys) = crate::mem::alloc_zeroed_page_phys() else {
                    return;
                };
                pdpt[i3] = pd_phys | PRESENT | WRITABLE;
            } else if (pdpt[i3] & HUGE) != 0 {
                let Some(pd_phys) = crate::mem::alloc_zeroed_page_phys() else {
                    return;
                };
                let pd = &mut *(pd_phys as *mut [u64; 512]);
                let gib_base = ((i4 as u64) << 39) | ((i3 as u64) << 30);
                for m in 0..512usize {
                    pd[m] = (gib_base | ((m as u64) << 21)) | PRESENT | WRITABLE | HUGE;
                }
                pdpt[i3] = pd_phys | PRESENT | WRITABLE;
            }

            let pd = &mut *((pdpt[i3] & ADDR_MASK) as *mut [u64; 512]);
            let phys_2m = addr;
            if (pd[i2] & PRESENT) == 0 {
                pd[i2] = phys_2m | PRESENT | WRITABLE | HUGE | (cache_bits & CACHE_MASK);
            } else if (pd[i2] & HUGE) != 0 {
                pd[i2] = (pd[i2] & !CACHE_MASK) | (cache_bits & CACHE_MASK);
            }
            addr += 2 << 20;
        }

        // Flush TLB by reloading CR3.
        let cr3: u64;
        asm!("mov {}, cr3", out(reg) cr3, options(nomem, nostack));
        asm!("mov cr3, {}", in(reg) cr3, options(nomem, nostack));
    }
}

pub const PAT_WC_BITS: u64 = 1 << 3; // PWT=1, PCD=0, PAT=0 -> PAT1 = WC
pub const PAT_UC_BITS: u64 = (1 << 3) | (1 << 4); // PWT=1, PCD=1, PAT=0 -> PAT3 = UC

static TSC_HZ: AtomicU64 = AtomicU64::new(2_500_000_000);
static BOOT_TSC: AtomicU64 = AtomicU64::new(0);

fn calibrate_tsc_pit() -> u64 {
    // Use PIT channel 2 one-shot for ~10 ms (11932 ticks at 1_193_182 Hz).
    const PIT_HZ: u64 = 1_193_182;
    const TICKS: u16 = 11_932; // ~10.000 ms
    unsafe {
        let gate = inb(0x61);
        outb(0x61, (gate & !0x02) | 0x01); // Enable ch2 gate, disable speaker
        outb(0x43, 0xB0); // Channel 2, lobyte/hibyte, mode 0 (interrupt on terminal count)
        outb(0x42, (TICKS & 0xFF) as u8);
        outb(0x42, (TICKS >> 8) as u8);

        let start = _rdtsc();
        while (inb(0x61) & 0x20) == 0 {
            core::hint::spin_loop();
        }
        let end = _rdtsc();
        let delta = end.saturating_sub(start);
        ((delta as u128) * (PIT_HZ as u128) / (TICKS as u128)) as u64
    }
}

pub fn calibrate_tsc() -> u64 {
    let max_leaf = __cpuid(0).eax;
    let mut hz = 0u64;

    // Leaf 0x15: exact TSC / core crystal clock ratio (Alder Lake i5-12400 provides this).
    if max_leaf >= 0x15 {
        let r = __cpuid(0x15);
        if r.eax != 0 && r.ebx != 0 && r.ecx != 0 {
            hz = (r.ecx as u64) * (r.ebx as u64) / (r.eax as u64);
        }
    }
    // Leaf 0x16: processor base frequency in MHz.
    if hz == 0 && max_leaf >= 0x16 {
        let r = __cpuid(0x16);
        let base_mhz = (r.eax & 0xFFFF) as u64;
        if base_mhz >= 400 && base_mhz <= 10_000 {
            hz = base_mhz * 1_000_000;
        }
    }
    // Fallback: calibrate directly against PIT channel 2.
    if hz < 100_000_000 {
        let pit_hz = calibrate_tsc_pit();
        if pit_hz >= 100_000_000 {
            hz = pit_hz;
        } else {
            hz = 2_500_000_000;
        }
    }

    TSC_HZ.store(hz, Ordering::Relaxed);
    BOOT_TSC.store(unsafe { _rdtsc() }, Ordering::Relaxed);
    hz
}

#[inline]
pub fn tsc_hz() -> u64 {
    TSC_HZ.load(Ordering::Relaxed)
}

#[inline]
pub fn now_nanos() -> u64 {
    let now = unsafe { _rdtsc() };
    let boot = BOOT_TSC.load(Ordering::Relaxed);
    let delta = now.saturating_sub(boot) as u128;
    let hz = TSC_HZ.load(Ordering::Relaxed).max(1) as u128;
    ((delta * 1_000_000_000u128) / hz) as u64
}

pub fn busy_wait_us(us: u64) {
    let target = now_nanos().saturating_add(us * 1_000);
    while now_nanos() < target {
        core::hint::spin_loop();
    }
}
