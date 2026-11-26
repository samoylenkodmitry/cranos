use core::arch::asm;

pub const KERNEL_CS: u16 = 0x08;
pub const KERNEL_DS: u16 = 0x10;
const TSS_SELECTOR: u16 = 0x18;

#[repr(C, packed)]
struct DescriptorTablePointer {
    limit: u16,
    base: u64,
}

#[repr(C, packed)]
struct TaskStateSegment {
    reserved1: u32,
    rsp: [u64; 3],
    reserved2: u64,
    ist: [u64; 7],
    reserved3: u64,
    reserved4: u16,
    iomap_base: u16,
}

static mut DOUBLE_FAULT_STACK: [u8; 32 * 1024] = [0; 32 * 1024];

static mut TSS: TaskStateSegment = TaskStateSegment {
    reserved1: 0,
    rsp: [0; 3],
    reserved2: 0,
    ist: [0; 7],
    reserved3: 0,
    reserved4: 0,
    iomap_base: core::mem::size_of::<TaskStateSegment>() as u16,
};

// Null, Kernel Code 64, Kernel Data 64, TSS (16 bytes = 2 slots)
static mut GDT: [u64; 5] = [
    0,
    0x00AF_9A00_0000_FFFF, // 64-bit kernel code (L=1, P=1, DPL=0, S=1, Type=0xA)
    0x00CF_9200_0000_FFFF, // 64-bit kernel data (P=1, DPL=0, S=1, Type=0x2)
    0,
    0,
];

pub fn init() {
    unsafe {
        let df_top = (&raw const DOUBLE_FAULT_STACK as *const u8 as u64) + 32 * 1024;
        TSS.ist[0] = df_top;

        let tss_addr = &raw const TSS as u64;
        let tss_limit = (core::mem::size_of::<TaskStateSegment>() - 1) as u64;
        let low = (tss_limit & 0xFFFF)
            | ((tss_addr & 0x00FF_FFFF) << 16)
            | (0x89u64 << 40) // Present, 64-bit available TSS
            | (((tss_limit >> 16) & 0xF) << 48)
            | (((tss_addr >> 24) & 0xFF) << 56);
        let high = tss_addr >> 32;
        GDT[3] = low;
        GDT[4] = high;

        let gdtr = DescriptorTablePointer {
            limit: (core::mem::size_of_val(&*(&raw const GDT)) - 1) as u16,
            base: &raw const GDT as u64,
        };

        asm!(
            "lgdt [{gdtr}]",
            "push {cs}",
            "lea rax, [rip + 2f]",
            "push rax",
            "retfq",
            "2:",
            "mov ax, {ds}",
            "mov ds, ax",
            "mov es, ax",
            "mov ss, ax",
            "xor ax, ax",
            "mov fs, ax",
            "mov gs, ax",
            "ltr {tss:x}",
            gdtr = in(reg) &gdtr,
            cs = const KERNEL_CS as u64,
            ds = const KERNEL_DS,
            tss = in(reg) TSS_SELECTOR,
            out("rax") _,
        );
    }
}

/// Loads the shared GDT (without TSS reload) on an Application Processor.
pub fn load_ap() {
    unsafe {
        let gdtr = DescriptorTablePointer {
            limit: (core::mem::size_of_val(&*(&raw const GDT)) - 1) as u16,
            base: &raw const GDT as u64,
        };
        asm!(
            "lgdt [{gdtr}]",
            "push {cs}",
            "lea rax, [rip + 2f]",
            "push rax",
            "retfq",
            "2:",
            "mov ax, {ds}",
            "mov ds, ax",
            "mov es, ax",
            "mov ss, ax",
            "xor ax, ax",
            "mov fs, ax",
            "mov gs, ax",
            gdtr = in(reg) &gdtr,
            cs = const KERNEL_CS as u64,
            ds = const KERNEL_DS,
            out("rax") _,
        );
    }
}
