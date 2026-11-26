use core::arch::asm;
use core::fmt::{self, Write};
use core::sync::atomic::{AtomicBool, Ordering};

const COM1: u16 = 0x3F8;

static SERIAL_LOCK: AtomicBool = AtomicBool::new(false);

#[inline]
pub unsafe fn outb(port: u16, val: u8) {
    unsafe { asm!("out dx, al", in("dx") port, in("al") val, options(nomem, nostack, preserves_flags)) };
}

#[inline]
pub unsafe fn inb(port: u16) -> u8 {
    let val: u8;
    unsafe { asm!("in al, dx", in("dx") port, out("al") val, options(nomem, nostack, preserves_flags)) };
    val
}

pub fn init() {
    unsafe {
        outb(COM1 + 1, 0x00); // Disable interrupts
        outb(COM1 + 3, 0x80); // Enable DLAB
        outb(COM1 + 0, 0x01); // 115200 baud divisor low
        outb(COM1 + 1, 0x00); // divisor high
        outb(COM1 + 3, 0x03); // 8N1
        outb(COM1 + 2, 0xC7); // Enable FIFO
        outb(COM1 + 4, 0x0B); // IRQs enabled, RTS/DSR set
    }
}

fn write_byte_raw(b: u8) {
    unsafe {
        for _ in 0..100_000 {
            if (inb(COM1 + 5) & 0x20) != 0 {
                break;
            }
            core::hint::spin_loop();
        }
        outb(COM1, b);
    }
}

pub fn try_read_byte() -> Option<u8> {
    unsafe {
        if (inb(COM1 + 5) & 0x01) != 0 {
            Some(inb(COM1))
        } else {
            None
        }
    }
}

pub struct SerialWriter;

impl Write for SerialWriter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for b in s.bytes() {
            if b == b'\n' {
                write_byte_raw(b'\r');
            }
            write_byte_raw(b);
        }
        Ok(())
    }
}

pub fn print_fmt(args: fmt::Arguments<'_>) {
    while SERIAL_LOCK
        .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        core::hint::spin_loop();
    }
    let _ = SerialWriter.write_fmt(args);
    SERIAL_LOCK.store(false, Ordering::Release);
}

#[macro_export]
macro_rules! kprint {
    ($($arg:tt)*) => {
        $crate::serial::print_fmt(format_args!($($arg)*))
    };
}

#[macro_export]
macro_rules! kprintln {
    () => {
        $crate::kprint!("\n")
    };
    ($($arg:tt)*) => {
        $crate::serial::print_fmt(format_args!("{}\n", format_args!($($arg)*)))
    };
}
