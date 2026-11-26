use core::arch::{asm, naked_asm};

use crate::gdt::KERNEL_CS;
use crate::kprintln;

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct IdtEntry {
    offset_lo: u16,
    selector: u16,
    ist: u8,
    type_attr: u8,
    offset_mid: u16,
    offset_hi: u32,
    zero: u32,
}

impl IdtEntry {
    const fn missing() -> Self {
        Self {
            offset_lo: 0,
            selector: 0,
            ist: 0,
            type_attr: 0,
            offset_mid: 0,
            offset_hi: 0,
            zero: 0,
        }
    }

    fn interrupt_gate(handler: u64, ist: u8) -> Self {
        Self {
            offset_lo: handler as u16,
            selector: KERNEL_CS,
            ist,
            type_attr: 0x8E, // Present, DPL=0, 64-bit Interrupt Gate
            offset_mid: (handler >> 16) as u16,
            offset_hi: (handler >> 32) as u32,
            zero: 0,
        }
    }
}

#[repr(C, packed)]
struct IdtPointer {
    limit: u16,
    base: u64,
}

#[repr(C)]
pub struct ExceptionFrame {
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub r11: u64,
    pub r10: u64,
    pub r9: u64,
    pub r8: u64,
    pub rbp: u64,
    pub rdi: u64,
    pub rsi: u64,
    pub rdx: u64,
    pub rcx: u64,
    pub rbx: u64,
    pub rax: u64,
    pub vector: u64,
    pub error_code: u64,
    pub rip: u64,
    pub cs: u64,
    pub rflags: u64,
    pub rsp: u64,
    pub ss: u64,
}

static mut IDT: [IdtEntry; 256] = [IdtEntry::missing(); 256];

extern "C" fn exception_dispatch(frame: &ExceptionFrame) -> ! {
    let cr2: u64;
    unsafe { asm!("mov {}, cr2", out(reg) cr2, options(nomem, nostack)) };
    let name = match frame.vector {
        0 => "#DE Divide Error",
        1 => "#DB Debug",
        2 => "NMI",
        3 => "#BP Breakpoint",
        6 => "#UD Invalid Opcode",
        7 => "#NM Device Not Available",
        8 => "#DF Double Fault",
        10 => "#TS Invalid TSS",
        11 => "#NP Segment Not Present",
        12 => "#SS Stack-Segment Fault",
        13 => "#GP General Protection Fault",
        14 => "#PF Page Fault",
        16 => "#MF x87 FPU Error",
        19 => "#XM SIMD Floating-Point",
        _ => "Exception",
    };
    kprintln!("\n=== CPU EXCEPTION: {} (vec={}) ===", name, frame.vector);
    kprintln!(
        "  RIP={:#018x} CS={:#x} RFLAGS={:#018x} ERR={:#x}",
        frame.rip,
        frame.cs,
        frame.rflags,
        frame.error_code
    );
    kprintln!("  RSP={:#018x} RBP={:#018x} CR2={:#018x}", frame.rsp, frame.rbp, cr2);
    kprintln!(
        "  RAX={:#018x} RBX={:#018x} RCX={:#018x} RDX={:#018x}",
        frame.rax,
        frame.rbx,
        frame.rcx,
        frame.rdx
    );
    kprintln!(
        "  RSI={:#018x} RDI={:#018x} R8 ={:#018x} R9 ={:#018x}",
        frame.rsi,
        frame.rdi,
        frame.r8,
        frame.r9
    );
    // Paint red banner across top of GOP framebuffer (with vector code stripes)
    // and auto-reboot after 10 seconds so headless/KVM hardware always recovers.
    crate::fb::diag_strip(8, 0x00E1_1D48, 24);
    crate::fb::diag_strip((frame.vector as u32 & 7).max(1), 0x00FA_CC15, 8);
    kprintln!("Auto-rebooting in 10s...");
    crate::cpu::busy_wait_us(10_000_000);
    crate::cpu::reboot();
}

#[unsafe(naked)]
extern "C" fn common_exception_stub() {
    naked_asm!(
        "push rax",
        "push rbx",
        "push rcx",
        "push rdx",
        "push rsi",
        "push rdi",
        "push rbp",
        "push r8",
        "push r9",
        "push r10",
        "push r11",
        "push r12",
        "push r13",
        "push r14",
        "push r15",
        "mov rdi, rsp",
        "and rsp, -16",
        "call {handler}",
        handler = sym exception_dispatch,
    );
}

macro_rules! exc_no_err {
    ($name:ident, $vec:literal) => {
        #[unsafe(naked)]
        extern "C" fn $name() {
            naked_asm!(
                "push 0",
                "push {vec}",
                "jmp {common}",
                vec = const $vec,
                common = sym common_exception_stub,
            );
        }
    };
}

macro_rules! exc_with_err {
    ($name:ident, $vec:literal) => {
        #[unsafe(naked)]
        extern "C" fn $name() {
            naked_asm!(
                "push {vec}",
                "jmp {common}",
                vec = const $vec,
                common = sym common_exception_stub,
            );
        }
    };
}

exc_no_err!(isr_de, 0);
exc_no_err!(isr_db, 1);
exc_no_err!(isr_nmi, 2);
exc_no_err!(isr_bp, 3);
exc_no_err!(isr_ud, 6);
exc_no_err!(isr_nm, 7);
exc_with_err!(isr_df, 8);
exc_with_err!(isr_ts, 10);
exc_with_err!(isr_np, 11);
exc_with_err!(isr_ss, 12);
exc_with_err!(isr_gp, 13);
exc_with_err!(isr_pf, 14);
exc_no_err!(isr_mf, 16);
exc_no_err!(isr_xm, 19);

#[unsafe(naked)]
extern "C" fn isr_spurious() {
    naked_asm!("iretq");
}

pub fn init() {
    unsafe {
        IDT[0] = IdtEntry::interrupt_gate(isr_de as *const () as u64, 0);
        IDT[1] = IdtEntry::interrupt_gate(isr_db as *const () as u64, 0);
        IDT[2] = IdtEntry::interrupt_gate(isr_nmi as *const () as u64, 0);
        IDT[3] = IdtEntry::interrupt_gate(isr_bp as *const () as u64, 0);
        IDT[6] = IdtEntry::interrupt_gate(isr_ud as *const () as u64, 0);
        IDT[7] = IdtEntry::interrupt_gate(isr_nm as *const () as u64, 0);
        IDT[8] = IdtEntry::interrupt_gate(isr_df as *const () as u64, 1); // IST1
        IDT[10] = IdtEntry::interrupt_gate(isr_ts as *const () as u64, 0);
        IDT[11] = IdtEntry::interrupt_gate(isr_np as *const () as u64, 0);
        IDT[12] = IdtEntry::interrupt_gate(isr_ss as *const () as u64, 0);
        IDT[13] = IdtEntry::interrupt_gate(isr_gp as *const () as u64, 0);
        IDT[14] = IdtEntry::interrupt_gate(isr_pf as *const () as u64, 0);
        IDT[16] = IdtEntry::interrupt_gate(isr_mf as *const () as u64, 0);
        IDT[19] = IdtEntry::interrupt_gate(isr_xm as *const () as u64, 0);
        IDT[0xFF] = IdtEntry::interrupt_gate(isr_spurious as *const () as u64, 0);
        load();
    }
}

pub fn load() {
    unsafe {
        let idtr = IdtPointer {
            limit: (core::mem::size_of_val(&*(&raw const IDT)) - 1) as u16,
            base: &raw const IDT as u64,
        };
        asm!("lidt [{}]", in(reg) &idtr, options(readonly, nostack, preserves_flags));
    }
}
