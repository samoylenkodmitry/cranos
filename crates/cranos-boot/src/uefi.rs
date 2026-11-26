#![allow(dead_code)]

use core::ffi::c_void;

pub type Handle = *mut c_void;
pub type Status = usize;

pub const EFI_SUCCESS: Status = 0;
pub const EFI_BUFFER_TOO_SMALL: Status = (1usize << 63) | 5;
pub const EFI_NOT_FOUND: Status = (1usize << 63) | 14;

pub const ALLOCATE_ANY_PAGES: u32 = 0;
pub const ALLOCATE_MAX_ADDRESS: u32 = 1;
pub const ALLOCATE_ADDRESS: u32 = 2;

pub const EFI_LOADER_CODE: u32 = 1;
pub const EFI_LOADER_DATA: u32 = 2;
pub const EFI_BOOT_SERVICES_CODE: u32 = 3;
pub const EFI_BOOT_SERVICES_DATA: u32 = 4;
pub const EFI_CONVENTIONAL_MEMORY: u32 = 7;
pub const EFI_ACPI_RECLAIM_MEMORY: u32 = 9;
pub const EFI_MEMORY_MAPPED_IO: u32 = 11;

pub const BY_PROTOCOL: u32 = 2;

pub const FILE_MODE_READ: u64 = 1;

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Guid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

pub const GRAPHICS_OUTPUT_PROTOCOL_GUID: Guid = Guid {
    data1: 0x9042a9de,
    data2: 0x23dc,
    data3: 0x4a38,
    data4: [0x96, 0xfb, 0x7a, 0xde, 0xd0, 0x80, 0x51, 0x6a],
};

pub const SIMPLE_FILE_SYSTEM_PROTOCOL_GUID: Guid = Guid {
    data1: 0x0964e5b22,
    data2: 0x6459,
    data3: 0x11d2,
    data4: [0x8e, 0x39, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
};

pub const FILE_INFO_ID: Guid = Guid {
    data1: 0x09576e92,
    data2: 0x6d3f,
    data3: 0x11d2,
    data4: [0x8e, 0x39, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
};

pub const BLOCK_IO_PROTOCOL_GUID: Guid = Guid {
    data1: 0x964e5b21,
    data2: 0x6459,
    data3: 0x11d2,
    data4: [0x8e, 0x39, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
};

pub const ACPI_20_TABLE_GUID: Guid = Guid {
    data1: 0x8868e871,
    data2: 0xe4f1,
    data3: 0x11d3,
    data4: [0xbc, 0x22, 0x00, 0x80, 0xc7, 0x3c, 0x88, 0x81],
};

pub const ACPI_10_TABLE_GUID: Guid = Guid {
    data1: 0xeb9d2d30,
    data2: 0x2d88,
    data3: 0x11d3,
    data4: [0x9a, 0x16, 0x00, 0x90, 0x27, 0x3f, 0xc1, 0x4d],
};

#[repr(C)]
pub struct TableHeader {
    pub signature: u64,
    pub revision: u32,
    pub header_size: u32,
    pub crc32: u32,
    pub reserved: u32,
}

#[repr(C)]
pub struct ConfigurationTable {
    pub vendor_guid: Guid,
    pub vendor_table: *mut c_void,
}

#[repr(C)]
pub struct SimpleTextOutputProtocol {
    pub reset: extern "efiapi" fn(*mut Self, bool) -> Status,
    pub output_string: extern "efiapi" fn(*mut Self, *const u16) -> Status,
    pub test_string: *const c_void,
    pub query_mode: *const c_void,
    pub set_mode: *const c_void,
    pub set_attribute: *const c_void,
    pub clear_screen: extern "efiapi" fn(*mut Self) -> Status,
}

#[repr(C)]
pub struct SystemTable {
    pub hdr: TableHeader,
    pub firmware_vendor: *const u16,
    pub firmware_revision: u32,
    pub console_in_handle: Handle,
    pub con_in: *mut c_void,
    pub console_out_handle: Handle,
    pub con_out: *mut SimpleTextOutputProtocol,
    pub standard_error_handle: Handle,
    pub std_err: *mut SimpleTextOutputProtocol,
    pub runtime_services: *mut c_void,
    pub boot_services: *mut BootServices,
    pub number_of_table_entries: usize,
    pub configuration_table: *const ConfigurationTable,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MemoryDescriptor {
    pub ty: u32,
    pub pad: u32,
    pub physical_start: u64,
    pub virtual_start: u64,
    pub number_of_pages: u64,
    pub attribute: u64,
}

#[repr(C)]
pub struct BootServices {
    pub hdr: TableHeader,
    // Task Priority Services (2)
    pub raise_tpl: *const c_void,
    pub restore_tpl: *const c_void,
    // Memory Services (5)
    pub allocate_pages:
        extern "efiapi" fn(alloc_type: u32, mem_type: u32, pages: usize, memory: *mut u64) -> Status,
    pub free_pages: extern "efiapi" fn(memory: u64, pages: usize) -> Status,
    pub get_memory_map: extern "efiapi" fn(
        map_size: *mut usize,
        map: *mut MemoryDescriptor,
        map_key: *mut usize,
        descriptor_size: *mut usize,
        descriptor_version: *mut u32,
    ) -> Status,
    pub allocate_pool: extern "efiapi" fn(pool_type: u32, size: usize, buffer: *mut *mut c_void) -> Status,
    pub free_pool: extern "efiapi" fn(buffer: *mut c_void) -> Status,
    // Event & Timer Services (6)
    pub create_event: *const c_void,
    pub set_timer: *const c_void,
    pub wait_for_event: *const c_void,
    pub signal_event: *const c_void,
    pub close_event: *const c_void,
    pub check_event: *const c_void,
    // Protocol Handler Services (9)
    pub install_protocol_interface: *const c_void,
    pub reinstall_protocol_interface: *const c_void,
    pub uninstall_protocol_interface: *const c_void,
    pub handle_protocol: extern "efiapi" fn(handle: Handle, protocol: *const Guid, interface: *mut *mut c_void) -> Status,
    pub reserved: *const c_void,
    pub register_protocol_notify: *const c_void,
    pub locate_handle: *const c_void,
    pub locate_device_path: *const c_void,
    pub install_configuration_table: *const c_void,
    // Image Services (5)
    pub load_image: *const c_void,
    pub start_image: *const c_void,
    pub exit: *const c_void,
    pub unload_image: *const c_void,
    pub exit_boot_services: extern "efiapi" fn(image_handle: Handle, map_key: usize) -> Status,
    // Misc Services (3)
    pub get_next_monotonic_count: *const c_void,
    pub stall: extern "efiapi" fn(microseconds: usize) -> Status,
    pub set_watchdog_timer:
        extern "efiapi" fn(timeout: usize, watchdog_code: u64, data_size: usize, watchdog_data: *const u16) -> Status,
    // DriverSupport Services (2)
    pub connect_controller: *const c_void,
    pub disconnect_controller: *const c_void,
    // Open and Close Protocol Services (3)
    pub open_protocol: *const c_void,
    pub close_protocol: *const c_void,
    pub open_protocol_information: *const c_void,
    // Library Services (4)
    pub protocols_per_handle: *const c_void,
    pub locate_handle_buffer: extern "efiapi" fn(
        search_type: u32,
        protocol: *const Guid,
        search_key: *const c_void,
        no_handles: *mut usize,
        buffer: *mut *mut Handle,
    ) -> Status,
    pub locate_protocol:
        extern "efiapi" fn(protocol: *const Guid, registration: *mut c_void, interface: *mut *mut c_void) -> Status,
    pub install_multiple_protocol_interfaces: *const c_void,
    pub uninstall_multiple_protocol_interfaces: *const c_void,
    // CRC32 + CopyMem + SetMem (3)
    pub calculate_crc32: *const c_void,
    pub copy_mem: *const c_void,
    pub set_mem: *const c_void,
    pub create_event_ex: *const c_void,
}

#[repr(C)]
pub struct GraphicsOutputModeInformation {
    pub version: u32,
    pub horizontal_resolution: u32,
    pub vertical_resolution: u32,
    pub pixel_format: u32, // 0=RGBX8, 1=BGRX8, 2=BitMask, 3=BltOnly
    pub pixel_information: [u32; 4],
    pub pixels_per_scan_line: u32,
}

#[repr(C)]
pub struct GraphicsOutputProtocolMode {
    pub max_mode: u32,
    pub mode: u32,
    pub info: *const GraphicsOutputModeInformation,
    pub size_of_info: usize,
    pub frame_buffer_base: u64,
    pub frame_buffer_size: usize,
}

#[repr(C)]
pub struct GraphicsOutputProtocol {
    pub query_mode: extern "efiapi" fn(
        *mut Self,
        mode_number: u32,
        size_of_info: *mut usize,
        info: *mut *const GraphicsOutputModeInformation,
    ) -> Status,
    pub set_mode: extern "efiapi" fn(*mut Self, mode_number: u32) -> Status,
    pub blt: *const c_void,
    pub mode: *const GraphicsOutputProtocolMode,
}

#[repr(C)]
pub struct SimpleFileSystemProtocol {
    pub revision: u64,
    pub open_volume: extern "efiapi" fn(*mut Self, root: *mut *mut FileProtocol) -> Status,
}

#[repr(C)]
pub struct FileProtocol {
    pub revision: u64,
    pub open: extern "efiapi" fn(
        *mut Self,
        new_handle: *mut *mut FileProtocol,
        file_name: *const u16,
        open_mode: u64,
        attributes: u64,
    ) -> Status,
    pub close: extern "efiapi" fn(*mut Self) -> Status,
    pub delete: *const c_void,
    pub read: extern "efiapi" fn(*mut Self, buffer_size: *mut usize, buffer: *mut c_void) -> Status,
    pub write: *const c_void,
    pub get_position: *const c_void,
    pub set_position: extern "efiapi" fn(*mut Self, position: u64) -> Status,
    pub get_info: extern "efiapi" fn(
        *mut Self,
        information_type: *const Guid,
        buffer_size: *mut usize,
        buffer: *mut c_void,
    ) -> Status,
}

#[repr(C)]
pub struct FileInfoHeader {
    pub size: u64,
    pub file_size: u64,
    pub physical_size: u64,
    pub create_time: [u8; 16],
    pub last_access_time: [u8; 16],
    pub modification_time: [u8; 16],
    pub attribute: u64,
}

#[repr(C)]
pub struct BlockIoMedia {
    pub media_id: u32,
    pub removable_media: bool,
    pub media_present: bool,
    pub logical_partition: bool,
    pub read_only: bool,
    pub write_caching: bool,
    pub block_size: u32,
    pub io_align: u32,
    pub last_block: u64,
}

#[repr(C)]
pub struct BlockIoProtocol {
    pub revision: u64,
    pub media: *const BlockIoMedia,
    pub reset: extern "efiapi" fn(*mut Self, extended_verification: bool) -> Status,
    pub read_blocks: extern "efiapi" fn(
        *mut Self,
        media_id: u32,
        lba: u64,
        buffer_size: usize,
        buffer: *mut c_void,
    ) -> Status,
}

pub fn print_ascii(st: &SystemTable, msg: &str) {
    let mut buf = [0u16; 128];
    let mut i = 0;
    for b in msg.bytes() {
        if b == b'\n' {
            if i + 2 >= buf.len() {
                buf[i] = 0;
                unsafe { ((*st.con_out).output_string)(st.con_out, buf.as_ptr()) };
                i = 0;
            }
            buf[i] = b'\r' as u16;
            i += 1;
            buf[i] = b'\n' as u16;
            i += 1;
        } else {
            if i + 1 >= buf.len() {
                buf[i] = 0;
                unsafe { ((*st.con_out).output_string)(st.con_out, buf.as_ptr()) };
                i = 0;
            }
            buf[i] = b as u16;
            i += 1;
        }
    }
    if i > 0 {
        buf[i] = 0;
        unsafe { ((*st.con_out).output_string)(st.con_out, buf.as_ptr()) };
    }
}

pub fn print_u64(st: &SystemTable, mut val: u64) {
    let mut tmp = [0u8; 24];
    let mut i = 0;
    if val == 0 {
        print_ascii(st, "0");
        return;
    }
    while val > 0 {
        tmp[i] = b'0' + (val % 10) as u8;
        val /= 10;
        i += 1;
    }
    let mut out = [0u8; 24];
    for j in 0..i {
        out[j] = tmp[i - 1 - j];
    }
    if let Ok(s) = core::str::from_utf8(&out[..i]) {
        print_ascii(st, s);
    }
}

pub fn print_hex(st: &SystemTable, val: u64) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = [0u8; 18];
    out[0] = b'0';
    out[1] = b'x';
    for i in 0..16 {
        let shift = (15 - i) * 4;
        out[2 + i] = HEX[((val >> shift) & 0xF) as usize];
    }
    if let Ok(s) = core::str::from_utf8(&out) {
        print_ascii(st, s);
    }
}
