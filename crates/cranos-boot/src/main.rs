#![no_std]
#![no_main]

mod elf;
mod ext4;
mod paging;
mod uefi;

use core::arch::asm;
use core::ffi::c_void;
use core::panic::PanicInfo;

use cranos_boot_info::{
    BOOT_INFO_MAGIC, BootInfo, FramebufferInfo, KERNEL_STACK_TOP, MemoryKind, MemoryRegion,
    PixelFormat,
};

use uefi::{
    ACPI_10_TABLE_GUID, ACPI_20_TABLE_GUID, ALLOCATE_ANY_PAGES, BLOCK_IO_PROTOCOL_GUID,
    BY_PROTOCOL, BlockIoProtocol, BootServices, EFI_ACPI_RECLAIM_MEMORY, EFI_BOOT_SERVICES_CODE,
    EFI_BOOT_SERVICES_DATA, EFI_BUFFER_TOO_SMALL, EFI_CONVENTIONAL_MEMORY, EFI_LOADER_CODE,
    EFI_LOADER_DATA, EFI_MEMORY_MAPPED_IO, EFI_SUCCESS, FILE_INFO_ID, FILE_MODE_READ,
    FileInfoHeader, FileProtocol, GRAPHICS_OUTPUT_PROTOCOL_GUID, GraphicsOutputProtocol, Handle,
    MemoryDescriptor, SIMPLE_FILE_SYSTEM_PROTOCOL_GUID, SimpleFileSystemProtocol, Status,
    SystemTable, print_ascii, print_hex, print_u64,
};

// Embedded higher-half bare-metal kernel ELF (set by scripts/build.sh).
static KERNEL_ELF: &[u8] = include_bytes!(env!("CRANOS_KERNEL_ELF"));

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {
        unsafe { asm!("hlt") };
    }
}

fn find_acpi_rsdp(st: &SystemTable) -> u64 {
    let tables = unsafe {
        core::slice::from_raw_parts(st.configuration_table, st.number_of_table_entries)
    };
    for t in tables {
        if t.vendor_guid == ACPI_20_TABLE_GUID {
            return t.vendor_table as u64;
        }
    }
    for t in tables {
        if t.vendor_guid == ACPI_10_TABLE_GUID {
            return t.vendor_table as u64;
        }
    }
    0
}

fn init_gop(bs: &BootServices, st: &SystemTable) -> Result<FramebufferInfo, &'static str> {
    let mut gop_ptr: *mut c_void = core::ptr::null_mut();
    let status = (bs.locate_protocol)(
        &GRAPHICS_OUTPUT_PROTOCOL_GUID,
        core::ptr::null_mut(),
        &mut gop_ptr,
    );
    if status != EFI_SUCCESS || gop_ptr.is_null() {
        return Err("GOP protocol not found");
    }
    let gop = gop_ptr as *mut GraphicsOutputProtocol;
    let mode_ptr = unsafe { (*gop).mode };
    let max_mode = unsafe { (*mode_ptr).max_mode };

    // Choose best 32-bit linear mode: prefer 1280x800 or 1280x720 or 1920x1080
    // (1280x800/1280x720/1920x1080 gives crisp UI and fast CPU blitting).
    let mut best_mode: Option<u32> = None;
    let mut best_score: i64 = -1;

    for m in 0..max_mode {
        let mut info_size = 0usize;
        let mut info_ptr: *const uefi::GraphicsOutputModeInformation = core::ptr::null();
        let qs = unsafe { ((*gop).query_mode)(gop, m, &mut info_size, &mut info_ptr) };
        if qs != EFI_SUCCESS || info_ptr.is_null() {
            continue;
        }
        let info = unsafe { &*info_ptr };
        if info.pixel_format != 0 && info.pixel_format != 1 {
            continue;
        }
        let w = info.horizontal_resolution;
        let h = info.vertical_resolution;
        if w < 800 || h < 600 || w > 2560 || h > 1600 {
            continue;
        }
        let score: i64 = if (w, h) == (1280, 800) || (w, h) == (1280, 720) {
            10_000_000
        } else if (w, h) == (1920, 1080) {
            9_500_000
        } else if (w, h) == (1280, 1024) {
            9_000_000
        } else if (w, h) == (1024, 768) {
            8_000_000
        } else {
            (w as i64) * (h as i64)
        };
        if score > best_score {
            best_score = score;
            best_mode = Some(m);
        }
    }

    if let Some(m) = best_mode {
        let cur = unsafe { (*(*gop).mode).mode };
        if m != cur {
            let _ = unsafe { ((*gop).set_mode)(gop, m) };
        }
    }

    let mode = unsafe { &*(*gop).mode };
    let info = unsafe { &*mode.info };
    let format = match info.pixel_format {
        0 => PixelFormat::Rgbx8,
        _ => PixelFormat::Bgrx8,
    };

    print_ascii(st, "[cranos-boot] GOP ");
    print_u64(st, info.horizontal_resolution as u64);
    print_ascii(st, "x");
    print_u64(st, info.vertical_resolution as u64);
    print_ascii(st, " fb=");
    print_hex(st, mode.frame_buffer_base);
    print_ascii(st, "\n");

    Ok(FramebufferInfo {
        phys_addr: mode.frame_buffer_base,
        size: mode.frame_buffer_size as u64,
        width: info.horizontal_resolution,
        height: info.vertical_resolution,
        stride_pixels: info.pixels_per_scan_line,
        format,
    })
}

struct LoadedModel {
    phys_addr: u64,
    size: u64,
    name: [u8; 64],
    name_len: u32,
}

fn ascii_to_utf16(s: &str, out: &mut [u16; 64]) -> *const u16 {
    let mut i = 0usize;
    for b in s.bytes() {
        if i + 1 >= out.len() {
            break;
        }
        out[i] = b as u16;
        i += 1;
    }
    out[i] = 0;
    out.as_ptr()
}

fn try_load_fat_model(bs: &BootServices, st: &SystemTable) -> Option<LoadedModel> {
    let mut handle_count = 0usize;
    let mut handles: *mut Handle = core::ptr::null_mut();
    let status = (bs.locate_handle_buffer)(
        BY_PROTOCOL,
        &SIMPLE_FILE_SYSTEM_PROTOCOL_GUID,
        core::ptr::null(),
        &mut handle_count,
        &mut handles,
    );
    if status != EFI_SUCCESS || handles.is_null() || handle_count == 0 {
        return None;
    }

    const CANDIDATES: &[&str] = &[
        "\\model.gguf",
        "\\models\\model.gguf",
        "\\cranos\\model.gguf",
    ];

    let mut loaded: Option<LoadedModel> = None;
    let handle_slice = unsafe { core::slice::from_raw_parts(handles, handle_count) };

    'outer: for &h in handle_slice {
        let mut sfs_ptr: *mut c_void = core::ptr::null_mut();
        if (bs.handle_protocol)(h, &SIMPLE_FILE_SYSTEM_PROTOCOL_GUID, &mut sfs_ptr) != EFI_SUCCESS {
            continue;
        }
        let sfs = sfs_ptr as *mut SimpleFileSystemProtocol;
        let mut root: *mut FileProtocol = core::ptr::null_mut();
        if unsafe { ((*sfs).open_volume)(sfs, &mut root) } != EFI_SUCCESS || root.is_null() {
            continue;
        }

        for &cand in CANDIDATES {
            let mut u16_name = [0u16; 64];
            let name_ptr = ascii_to_utf16(cand, &mut u16_name);
            let mut file: *mut FileProtocol = core::ptr::null_mut();
            let open_st = unsafe { ((*root).open)(root, &mut file, name_ptr, FILE_MODE_READ, 0) };
            if open_st != EFI_SUCCESS || file.is_null() {
                continue;
            }

            let mut info_buf = [0u8; 256];
            let mut info_size = info_buf.len();
            let gi_st = unsafe {
                ((*file).get_info)(
                    file,
                    &FILE_INFO_ID,
                    &mut info_size,
                    info_buf.as_mut_ptr() as *mut c_void,
                )
            };
            if gi_st != EFI_SUCCESS {
                unsafe { ((*file).close)(file) };
                continue;
            }
            let file_info = unsafe { core::ptr::read_unaligned(info_buf.as_ptr() as *const FileInfoHeader) };
            let file_size = file_info.file_size;
            if file_size == 0 {
                unsafe { ((*file).close)(file) };
                continue;
            }

            print_ascii(st, "[cranos-boot] Loading FAT ");
            print_ascii(st, cand);
            print_ascii(st, " (");
            print_u64(st, file_size / (1024 * 1024));
            print_ascii(st, " MiB)...\n");

            // 2 MiB-aligned allocation for optimal huge-page TLB behavior.
            let align = 2 * 1024 * 1024u64;
            let pages = ((file_size + align + 4095) / 4096) as usize;
            let mut raw_addr = 0u64;
            if (bs.allocate_pages)(ALLOCATE_ANY_PAGES, EFI_LOADER_DATA, pages, &mut raw_addr)
                != EFI_SUCCESS
            {
                unsafe { ((*file).close)(file) };
                continue;
            }
            let aligned = (raw_addr + align - 1) & !(align - 1);

            let mut offset = 0usize;
            let mut ok = true;
            while offset < file_size as usize {
                let mut chunk = (file_size as usize - offset).min(16 * 1024 * 1024);
                let rd_st = unsafe {
                    ((*file).read)(file, &mut chunk, (aligned as *mut u8).add(offset) as *mut c_void)
                };
                if rd_st != EFI_SUCCESS || chunk == 0 {
                    ok = false;
                    break;
                }
                offset += chunk;
            }
            unsafe { ((*file).close)(file) };

            if ok {
                let mut name = [0u8; 64];
                let bytes = cand.as_bytes();
                let len = bytes.len().min(64);
                name[..len].copy_from_slice(&bytes[..len]);
                loaded = Some(LoadedModel {
                    phys_addr: aligned,
                    size: file_size,
                    name,
                    name_len: len as u32,
                });
                unsafe { ((*root).close)(root) };
                break 'outer;
            } else {
                let _ = (bs.free_pages)(raw_addr, pages);
            }
        }
        unsafe { ((*root).close)(root) };
    }

    let _ = (bs.free_pool)(handles as *mut c_void);
    loaded
}

fn try_load_ext4_model(bs: &BootServices, st: &SystemTable) -> Option<LoadedModel> {
    let mut handle_count = 0usize;
    let mut handles: *mut Handle = core::ptr::null_mut();
    let status = (bs.locate_handle_buffer)(
        BY_PROTOCOL,
        &BLOCK_IO_PROTOCOL_GUID,
        core::ptr::null(),
        &mut handle_count,
        &mut handles,
    );
    if status != EFI_SUCCESS || handles.is_null() || handle_count == 0 {
        return None;
    }

    const EXT4_PATHS: &[(&[&str], &str)] = &[
        (&["cranos", "model.gguf"], "ext4:/cranos/model.gguf"),
        (&["models", "model.gguf"], "ext4:/models/model.gguf"),
        (&["model.gguf"], "ext4:/model.gguf"),
    ];

    let mut loaded = None;
    let handle_slice = unsafe { core::slice::from_raw_parts(handles, handle_count) };
    'outer: for &h in handle_slice {
        let mut bio_ptr: *mut c_void = core::ptr::null_mut();
        if (bs.handle_protocol)(h, &BLOCK_IO_PROTOCOL_GUID, &mut bio_ptr) != EFI_SUCCESS {
            continue;
        }
        let bio = bio_ptr as *mut BlockIoProtocol;
        for &(parts, label) in EXT4_PATHS {
            if let Some(file) = ext4::try_load_ext4_file(bs, bio, parts) {
                print_ascii(st, "[cranos-boot] Loaded ");
                print_ascii(st, label);
                print_ascii(st, " (");
                print_u64(st, file.size / (1024 * 1024));
                print_ascii(st, " MiB)\n");
                let mut name = [0u8; 64];
                let len = label.len().min(64);
                name[..len].copy_from_slice(&label.as_bytes()[..len]);
                loaded = Some(LoadedModel {
                    phys_addr: file.phys_addr,
                    size: file.size,
                    name,
                    name_len: len as u32,
                });
                break 'outer;
            }
        }
    }

    let _ = (bs.free_pool)(handles as *mut c_void);
    loaded
}

fn enable_sse_avx() {
    unsafe {
        let mut cr0: u64;
        asm!("mov {}, cr0", out(reg) cr0, options(nomem, nostack));
        cr0 &= !((1 << 2) | (1 << 3)); // Clear EM and TS
        cr0 |= 1 << 1; // Set MP
        asm!("mov cr0, {}", in(reg) cr0, options(nomem, nostack));

        let mut cr4: u64;
        asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack));
        cr4 |= (1 << 9) | (1 << 10); // OSFXSR | OSXMMEXCPT

        let cpuid1 = core::arch::x86_64::__cpuid(1);
        let has_xsave = (cpuid1.ecx & (1 << 26)) != 0;
        let has_avx = (cpuid1.ecx & (1 << 28)) != 0;
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
            eax |= 0x7; // x87 + SSE + AVX
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

fn classify_efi_mem(ty: u32) -> MemoryKind {
    match ty {
        EFI_CONVENTIONAL_MEMORY => MemoryKind::Usable,
        EFI_BOOT_SERVICES_CODE | EFI_BOOT_SERVICES_DATA => MemoryKind::BootReclaimable,
        EFI_ACPI_RECLAIM_MEMORY => MemoryKind::AcpiReclaimable,
        EFI_MEMORY_MAPPED_IO => MemoryKind::Mmio,
        EFI_LOADER_CODE | EFI_LOADER_DATA => MemoryKind::Reserved,
        _ => MemoryKind::Reserved,
    }
}

#[unsafe(no_mangle)]
pub extern "efiapi" fn efi_main(image_handle: Handle, system_table: *mut SystemTable) -> Status {
    let st = unsafe { &*system_table };
    let bs = unsafe { &*st.boot_services };

    // Disable the 5-minute UEFI watchdog immediately.
    let _ = (bs.set_watchdog_timer)(0, 0, 0, core::ptr::null());
    print_ascii(st, "\n[cranos-boot] CranOS UEFI Stage 1\n");

    let fb = match init_gop(bs, st) {
        Ok(fb) => fb,
        Err(msg) => {
            print_ascii(st, "[cranos-boot] FATAL GOP: ");
            print_ascii(st, msg);
            print_ascii(st, "\n");
            loop {
                unsafe { asm!("hlt") };
            }
        }
    };

    let acpi_rsdp = find_acpi_rsdp(st);
    print_ascii(st, "[cranos-boot] ACPI RSDP=");
    print_hex(st, acpi_rsdp);
    print_ascii(st, "\n");

    let model = try_load_fat_model(bs, st).or_else(|| try_load_ext4_model(bs, st));
    if model.is_none() {
        print_ascii(st, "[cranos-boot] No model.gguf found on FAT32/ext4; booting UI only\n");
    }

    let kernel = match elf::load_kernel_elf(bs, KERNEL_ELF) {
        Ok(k) => k,
        Err(msg) => {
            print_ascii(st, "[cranos-boot] FATAL ELF: ");
            print_ascii(st, msg);
            print_ascii(st, "\n");
            loop {
                unsafe { asm!("hlt") };
            }
        }
    };
    print_ascii(st, "[cranos-boot] Kernel ELF entry=");
    print_hex(st, kernel.entry_virt);
    print_ascii(st, " phys=");
    print_hex(st, kernel.phys_base);
    print_ascii(st, " size=");
    print_u64(st, kernel.size);
    print_ascii(st, "\n");

    // Pre-allocate BootInfo, copied MemoryRegion table, and UEFI memory map scratch
    // BEFORE building page tables so everything lives in known physical memory.
    let mut boot_info_phys = 0u64;
    if (bs.allocate_pages)(ALLOCATE_ANY_PAGES, EFI_LOADER_DATA, 1, &mut boot_info_phys)
        != EFI_SUCCESS
    {
        loop {
            unsafe { asm!("hlt") };
        }
    }
    let mut regions_phys = 0u64;
    let max_regions = 1024usize;
    let regions_pages = (max_regions * core::mem::size_of::<MemoryRegion>() + 4095) / 4096;
    if (bs.allocate_pages)(
        ALLOCATE_ANY_PAGES,
        EFI_LOADER_DATA,
        regions_pages,
        &mut regions_phys,
    ) != EFI_SUCCESS
    {
        loop {
            unsafe { asm!("hlt") };
        }
    }

    let mut mmap_buf_phys = 0u64;
    let mmap_buf_pages = 32usize; // 128 KiB
    if (bs.allocate_pages)(
        ALLOCATE_ANY_PAGES,
        EFI_LOADER_DATA,
        mmap_buf_pages,
        &mut mmap_buf_phys,
    ) != EFI_SUCCESS
    {
        loop {
            unsafe { asm!("hlt") };
        }
    }

    // Probe highest physical address first.
    let mut probe_size = mmap_buf_pages * 4096;
    let mut probe_key = 0usize;
    let mut desc_size = 0usize;
    let mut desc_ver = 0u32;
    let mut max_phys = fb.phys_addr.saturating_add(fb.size);
    if (bs.get_memory_map)(
        &mut probe_size,
        mmap_buf_phys as *mut MemoryDescriptor,
        &mut probe_key,
        &mut desc_size,
        &mut desc_ver,
    ) == EFI_SUCCESS
        && desc_size > 0
    {
        let count = probe_size / desc_size;
        for i in 0..count {
            let d = unsafe {
                &*((mmap_buf_phys as *const u8).add(i * desc_size) as *const MemoryDescriptor)
            };
            let end = d.physical_start.saturating_add(d.number_of_pages * 4096);
            if end > max_phys {
                max_phys = end;
            }
        }
    }

    let pt = match paging::build_page_tables(bs, &kernel, max_phys) {
        Ok(pt) => pt,
        Err(msg) => {
            print_ascii(st, "[cranos-boot] FATAL PAGING: ");
            print_ascii(st, msg);
            print_ascii(st, "\n");
            loop {
                unsafe { asm!("hlt") };
            }
        }
    };

    print_ascii(st, "[cranos-boot] Exiting UEFI Boot Services...\n");

    // Final GetMemoryMap + ExitBootServices retry loop (no allocations inside!).
    let mut region_count: usize;
    loop {
        let mut map_size = mmap_buf_pages * 4096;
        let mut map_key = 0usize;
        let mut dsize = 0usize;
        let mut dver = 0u32;
        let st_map = (bs.get_memory_map)(
            &mut map_size,
            mmap_buf_phys as *mut MemoryDescriptor,
            &mut map_key,
            &mut dsize,
            &mut dver,
        );
        if st_map == EFI_BUFFER_TOO_SMALL || st_map != EFI_SUCCESS || dsize == 0 {
            loop {
                unsafe { asm!("hlt") };
            }
        }

        let entries = map_size / dsize;
        let out_regions =
            unsafe { core::slice::from_raw_parts_mut(regions_phys as *mut MemoryRegion, max_regions) };
        region_count = 0;
        for i in 0..entries {
            if region_count >= max_regions {
                break;
            }
            let d = unsafe {
                &*((mmap_buf_phys as *const u8).add(i * dsize) as *const MemoryDescriptor)
            };
            let len = d.number_of_pages * 4096;
            if len == 0 {
                continue;
            }
            let kind = classify_efi_mem(d.ty);
            if region_count > 0 {
                let prev = &mut out_regions[region_count - 1];
                if prev.kind == kind && prev.start + prev.len == d.physical_start {
                    prev.len += len;
                    continue;
                }
            }
            out_regions[region_count] = MemoryRegion {
                start: d.physical_start,
                len,
                kind,
                _pad: 0,
            };
            region_count += 1;
        }

        let exit_st = (bs.exit_boot_services)(image_handle, map_key);
        if exit_st == EFI_SUCCESS {
            break;
        }
    }

    // Boot services are gone; we own the machine bare-metal from this instruction on.
    unsafe { asm!("cli", options(nomem, nostack)) };

    let (model_phys, model_size, model_name, model_name_len) = match model {
        Some(m) => (m.phys_addr, m.size, m.name, m.name_len),
        None => (0, 0, [0u8; 64], 0),
    };

    unsafe {
        *(boot_info_phys as *mut BootInfo) = BootInfo {
            magic: BOOT_INFO_MAGIC,
            fb,
            memory_map_ptr: regions_phys,
            memory_map_len: region_count as u64,
            acpi_rsdp_addr: acpi_rsdp,
            kernel_phys_base: kernel.phys_base,
            kernel_size: kernel.size,
            pml4_phys_addr: pt.pml4_phys,
            model_phys_addr: model_phys,
            model_size,
            model_name,
            model_name_len,
            model_load_ms: 0,
            has_1gib_pages: u8::from(pt.has_1gib_pages),
            _pad: [0; 7],
        };
    }

    enable_sse_avx();

    // Switch CR3 to our 1 GiB / 2 MiB huge-page table, pivot to the 1 MiB
    // higher-half kernel stack (aligned to 16n+8 to match `call` entry ABI),
    // and jump to kernel_main(boot_info_phys) in System V x86_64 ABI.
    unsafe {
        asm!(
            "mov cr3, {cr3}",
            "mov rsp, {stack}",
            "xor rbp, rbp",
            "jmp {entry}",
            cr3 = in(reg) pt.pml4_phys,
            stack = in(reg) KERNEL_STACK_TOP - 8,
            entry = in(reg) kernel.entry_virt,
            in("rdi") boot_info_phys,
            options(noreturn)
        );
    }
}
