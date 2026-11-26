use core::arch::x86_64::__cpuid;

use cranos_boot_info::{KERNEL_STACK_PAGES, KERNEL_STACK_TOP};

use crate::elf::LoadedKernel;
use crate::uefi::{ALLOCATE_ANY_PAGES, ALLOCATE_MAX_ADDRESS, BootServices, EFI_LOADER_DATA, EFI_SUCCESS};

const PRESENT: u64 = 1 << 0;
const WRITABLE: u64 = 1 << 1;
const HUGE: u64 = 1 << 7;
const ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;

fn alloc_zero_page(bs: &BootServices) -> Result<u64, &'static str> {
    // Keep PML4 and boot page tables strictly below 4 GiB so the 32-bit
    // protected-mode AP startup trampoline can load CR3 without truncation on
    // high-RAM machines (e.g. 64 GiB samarch).
    let mut addr = 0xFFFF_F000u64;
    let st = (bs.allocate_pages)(ALLOCATE_MAX_ADDRESS, EFI_LOADER_DATA, 1, &mut addr);
    if st != EFI_SUCCESS {
        return Err("failed to allocate page table page below 4 GiB");
    }
    unsafe {
        core::ptr::write_bytes(addr as *mut u8, 0, 4096);
    }
    Ok(addr)
}

unsafe fn table_mut(phys: u64) -> &'static mut [u64; 512] {
    unsafe { &mut *(phys as *mut [u64; 512]) }
}

fn ensure_table(bs: &BootServices, entry: &mut u64) -> Result<u64, &'static str> {
    if (*entry & PRESENT) != 0 && (*entry & HUGE) == 0 {
        return Ok(*entry & ADDR_MASK);
    }
    let next = alloc_zero_page(bs)?;
    *entry = next | PRESENT | WRITABLE;
    Ok(next)
}

fn map_4k(bs: &BootServices, pml4: u64, virt: u64, phys: u64, flags: u64) -> Result<(), &'static str> {
    let i4 = ((virt >> 39) & 0x1FF) as usize;
    let i3 = ((virt >> 30) & 0x1FF) as usize;
    let i2 = ((virt >> 21) & 0x1FF) as usize;
    let i1 = ((virt >> 12) & 0x1FF) as usize;

    let pdpt = ensure_table(bs, unsafe { &mut table_mut(pml4)[i4] })?;
    let pd = ensure_table(bs, unsafe { &mut table_mut(pdpt)[i3] })?;
    let pt = ensure_table(bs, unsafe { &mut table_mut(pd)[i2] })?;
    unsafe {
        table_mut(pt)[i1] = (phys & ADDR_MASK) | flags;
    }
    Ok(())
}

pub fn cpu_supports_1gib_pages() -> bool {
    let ext = __cpuid(0x8000_0000);
    if ext.eax < 0x8000_0001 {
        return false;
    }
    let feat = __cpuid(0x8000_0001);
    (feat.edx & (1 << 26)) != 0
}

pub struct BuiltPageTables {
    pub pml4_phys: u64,
    pub has_1gib_pages: bool,
}

pub fn build_page_tables(
    bs: &BootServices,
    kernel: &LoadedKernel,
    max_phys_addr: u64,
) -> Result<BuiltPageTables, &'static str> {
    let pml4 = alloc_zero_page(bs)?;
    let has_1gib = cpu_supports_1gib_pages();

    // Identity-map physical address space: when 1 GiB pages are available, map
    // the full 512 GiB slot 0 (costs just one 4 KiB PDPT page!) so high 64-bit
    // PCIe MMIO BARs (e.g. xHCI at 0x70_0000_0000) are always reachable.
    let map_gib = ((max_phys_addr.max(64 * 1024 * 1024 * 1024) + (1 << 30) - 1) >> 30).min(512);
    let id_pdpt = alloc_zero_page(bs)?;
    unsafe {
        table_mut(pml4)[0] = id_pdpt | PRESENT | WRITABLE;
    }

    if has_1gib {
        let pdpt_tbl = unsafe { table_mut(id_pdpt) };
        for g in 0..512usize {
            pdpt_tbl[g] = ((g as u64) << 30) | PRESENT | WRITABLE | HUGE;
        }
    } else {
        let map_gib_2m = map_gib.min(128);
        for g in 0..map_gib_2m as usize {
            let pd = alloc_zero_page(bs)?;
            unsafe {
                table_mut(id_pdpt)[g] = pd | PRESENT | WRITABLE;
            }
            let pd_tbl = unsafe { table_mut(pd) };
            for m in 0..512usize {
                let phys = ((g as u64) << 30) | ((m as u64) << 21);
                pd_tbl[m] = phys | PRESENT | WRITABLE | HUGE;
            }
        }
    }

    // Map higher-half kernel image at 4 KiB granularity.
    let kernel_pages = kernel.size / 4096;
    for p in 0..kernel_pages {
        let off = p * 4096;
        map_4k(
            bs,
            pml4,
            kernel.virt_base + off,
            kernel.phys_base + off,
            PRESENT | WRITABLE,
        )?;
    }

    // Allocate and map 1 MiB kernel stack below KERNEL_STACK_TOP (leaving a
    // guard page unmapped directly beneath the stack).
    let mut stack_phys = 0u64;
    let st = (bs.allocate_pages)(
        ALLOCATE_ANY_PAGES,
        EFI_LOADER_DATA,
        KERNEL_STACK_PAGES as usize,
        &mut stack_phys,
    );
    if st != EFI_SUCCESS {
        return Err("failed to allocate kernel stack");
    }
    unsafe {
        core::ptr::write_bytes(stack_phys as *mut u8, 0, (KERNEL_STACK_PAGES * 4096) as usize);
    }
    let stack_bottom = KERNEL_STACK_TOP - KERNEL_STACK_PAGES * 4096;
    for p in 0..KERNEL_STACK_PAGES {
        map_4k(
            bs,
            pml4,
            stack_bottom + p * 4096,
            stack_phys + p * 4096,
            PRESENT | WRITABLE,
        )?;
    }

    Ok(BuiltPageTables {
        pml4_phys: pml4,
        has_1gib_pages: has_1gib,
    })
}
