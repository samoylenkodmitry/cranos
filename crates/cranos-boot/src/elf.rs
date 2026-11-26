use crate::uefi::{ALLOCATE_ANY_PAGES, BootServices, EFI_LOADER_DATA, EFI_SUCCESS};

const ELF_MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];
const PT_LOAD: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
struct Elf64Header {
    ident: [u8; 16],
    ty: u16,
    machine: u16,
    version: u32,
    entry: u64,
    phoff: u64,
    shoff: u64,
    flags: u32,
    ehsize: u16,
    phentsize: u16,
    phnum: u16,
    shentsize: u16,
    shnum: u16,
    shstrndx: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Elf64ProgramHeader {
    ty: u32,
    flags: u32,
    offset: u64,
    vaddr: u64,
    paddr: u64,
    filesz: u64,
    memsz: u64,
    align: u64,
}

pub struct LoadedKernel {
    pub entry_virt: u64,
    pub virt_base: u64,
    pub phys_base: u64,
    pub size: u64,
}

pub fn load_kernel_elf(bs: &BootServices, elf: &[u8]) -> Result<LoadedKernel, &'static str> {
    if elf.len() < core::mem::size_of::<Elf64Header>() {
        return Err("kernel ELF too small");
    }
    let hdr = unsafe { core::ptr::read_unaligned(elf.as_ptr() as *const Elf64Header) };
    if hdr.ident[0..4] != ELF_MAGIC || hdr.ident[4] != 2 || hdr.machine != 0x3E {
        return Err("invalid x86_64 ELF header");
    }

    let mut min_vaddr = u64::MAX;
    let mut max_vaddr = 0u64;
    for i in 0..hdr.phnum as usize {
        let off = hdr.phoff as usize + i * hdr.phentsize as usize;
        if off + core::mem::size_of::<Elf64ProgramHeader>() > elf.len() {
            return Err("truncated program header");
        }
        let ph = unsafe {
            core::ptr::read_unaligned(elf.as_ptr().add(off) as *const Elf64ProgramHeader)
        };
        if ph.ty != PT_LOAD || ph.memsz == 0 {
            continue;
        }
        if ph.vaddr < min_vaddr {
            min_vaddr = ph.vaddr;
        }
        let end = ph.vaddr.checked_add(ph.memsz).ok_or("vaddr overflow")?;
        if end > max_vaddr {
            max_vaddr = end;
        }
    }
    if min_vaddr == u64::MAX || max_vaddr <= min_vaddr {
        return Err("no PT_LOAD segments");
    }

    let virt_base = min_vaddr & !0xFFF;
    let aligned_end = (max_vaddr + 0xFFF) & !0xFFF;
    let total_size = aligned_end - virt_base;
    let pages = (total_size / 4096) as usize;

    let mut phys_base = 0u64;
    let status = (bs.allocate_pages)(ALLOCATE_ANY_PAGES, EFI_LOADER_DATA, pages, &mut phys_base);
    if status != EFI_SUCCESS {
        return Err("failed to allocate kernel pages");
    }
    unsafe {
        core::ptr::write_bytes(phys_base as *mut u8, 0, total_size as usize);
    }

    for i in 0..hdr.phnum as usize {
        let off = hdr.phoff as usize + i * hdr.phentsize as usize;
        let ph = unsafe {
            core::ptr::read_unaligned(elf.as_ptr().add(off) as *const Elf64ProgramHeader)
        };
        if ph.ty != PT_LOAD || ph.filesz == 0 {
            continue;
        }
        let file_end = (ph.offset as usize)
            .checked_add(ph.filesz as usize)
            .ok_or("segment file overflow")?;
        if file_end > elf.len() {
            return Err("segment extends past ELF end");
        }
        let dst_off = (ph.vaddr - virt_base) as usize;
        unsafe {
            core::ptr::copy_nonoverlapping(
                elf.as_ptr().add(ph.offset as usize),
                (phys_base as *mut u8).add(dst_off),
                ph.filesz as usize,
            );
        }
    }

    Ok(LoadedKernel {
        entry_virt: hdr.entry,
        virt_base,
        phys_base,
        size: total_size,
    })
}
