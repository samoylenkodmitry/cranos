#![no_std]

pub const BOOT_INFO_MAGIC: u64 = 0x4352_414E_4F53_3031; // "CRANOS01"
pub const KERNEL_VIRT_BASE: u64 = 0xFFFF_FFFF_8000_0000;
pub const KERNEL_STACK_TOP: u64 = 0xFFFF_FFFF_9000_0000;
pub const KERNEL_STACK_PAGES: u64 = 256; // 1 MiB kernel stack for Cranpose + LLM

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Bgrx8 = 0,
    Rgbx8 = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FramebufferInfo {
    pub phys_addr: u64,
    pub size: u64,
    pub width: u32,
    pub height: u32,
    pub stride_pixels: u32,
    pub format: PixelFormat,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryKind {
    Usable = 0,
    Reserved = 1,
    AcpiReclaimable = 2,
    Mmio = 3,
    Kernel = 4,
    ModelWeights = 5,
    BootReclaimable = 6,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MemoryRegion {
    pub start: u64,
    pub len: u64,
    pub kind: MemoryKind,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BootInfo {
    pub magic: u64,
    pub fb: FramebufferInfo,
    pub memory_map_ptr: u64,
    pub memory_map_len: u64,
    pub acpi_rsdp_addr: u64,
    pub kernel_phys_base: u64,
    pub kernel_size: u64,
    pub pml4_phys_addr: u64,
    pub model_phys_addr: u64,
    pub model_size: u64,
    pub model_name: [u8; 64],
    pub model_name_len: u32,
    pub model_load_ms: u32,
    pub has_1gib_pages: u8,
    pub _pad: [u8; 7],
}

impl BootInfo {
    pub fn model_name_str(&self) -> &str {
        let len = (self.model_name_len as usize).min(self.model_name.len());
        core::str::from_utf8(&self.model_name[..len]).unwrap_or("")
    }

    /// # Safety
    /// Caller must ensure `memory_map_ptr` is identity-mapped and valid.
    pub unsafe fn memory_regions(&self) -> &[MemoryRegion] {
        if self.memory_map_ptr == 0 || self.memory_map_len == 0 {
            return &[];
        }
        unsafe {
            core::slice::from_raw_parts(
                self.memory_map_ptr as *const MemoryRegion,
                self.memory_map_len as usize,
            )
        }
    }
}
