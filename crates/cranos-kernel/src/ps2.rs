use crate::input::{self, InputEvent, KeyCode};
use crate::serial::{inb, outb};

const DATA_PORT: u16 = 0x60;
const STATUS_PORT: u16 = 0x64;

static mut PRESENT: bool = false;
static mut SHIFT: bool = false;
static mut CTRL: bool = false;
static mut ALT: bool = false;
static mut EXTENDED: bool = false;
static mut MOUSE_PHASE: u8 = 0;
static mut MOUSE_BYTES: [u8; 3] = [0; 3];
static mut MOUSE_BUTTONS: u8 = 0;

fn wait_write() -> bool {
    for _ in 0..10_000 {
        if (unsafe { inb(STATUS_PORT) } & 0x02) == 0 {
            return true;
        }
        core::hint::spin_loop();
    }
    false
}

fn wait_read() -> bool {
    for _ in 0..10_000 {
        if (unsafe { inb(STATUS_PORT) } & 0x01) != 0 {
            return true;
        }
        core::hint::spin_loop();
    }
    false
}

pub fn init() {
    let status = unsafe { inb(STATUS_PORT) };
    // No i8042 controller on modern USB-only systems (e.g. samarch-1 Intel 600-series).
    if status == 0xFF {
        unsafe { PRESENT = false };
        return;
    }

    unsafe {
        PRESENT = true;
        // Drain any stale bytes.
        for _ in 0..32 {
            if (inb(STATUS_PORT) & 0x01) == 0 {
                break;
            }
            let _ = inb(DATA_PORT);
        }

        // Try enabling auxiliary PS/2 mouse port for QEMU interactive testing.
        if wait_write() {
            outb(STATUS_PORT, 0xA8); // Enable aux port
        }
        if wait_write() {
            outb(STATUS_PORT, 0xD4); // Write to mouse
            if wait_write() {
                outb(DATA_PORT, 0xF4); // Enable data reporting
                if wait_read() {
                    let _ = inb(DATA_PORT); // ACK
                }
            }
        }
    }
}

fn scancode_set1(code: u8, shift: bool, extended: bool) -> Option<KeyCode> {
    if extended {
        return match code {
            0x48 => Some(KeyCode::Up),
            0x50 => Some(KeyCode::Down),
            0x4B => Some(KeyCode::Left),
            0x4D => Some(KeyCode::Right),
            0x47 => Some(KeyCode::Home),
            0x4F => Some(KeyCode::End),
            0x49 => Some(KeyCode::PageUp),
            0x51 => Some(KeyCode::PageDown),
            0x53 => Some(KeyCode::Delete),
            _ => None,
        };
    }

    match code {
        0x01 => Some(KeyCode::Escape),
        0x02..=0x0B => {
            const UNSHIFTED: [u8; 10] = *b"1234567890";
            const SHIFTED: [u8; 10] = *b"!@#$%^&*()";
            let i = (code - 0x02) as usize;
            Some(KeyCode::Char((if shift { SHIFTED[i] } else { UNSHIFTED[i] }) as char))
        }
        0x0C => Some(KeyCode::Char(if shift { '_' } else { '-' })),
        0x0D => Some(KeyCode::Char(if shift { '+' } else { '=' })),
        0x0E => Some(KeyCode::Backspace),
        0x0F => Some(KeyCode::Tab),
        0x10..=0x19 => {
            const ROW: [u8; 10] = *b"qwertyuiop";
            let c = ROW[(code - 0x10) as usize];
            Some(KeyCode::Char(if shift { c.to_ascii_uppercase() } else { c } as char))
        }
        0x1A => Some(KeyCode::Char(if shift { '{' } else { '[' })),
        0x1B => Some(KeyCode::Char(if shift { '}' } else { ']' })),
        0x1C => Some(KeyCode::Enter),
        0x1E..=0x26 => {
            const ROW: [u8; 9] = *b"asdfghjkl";
            let c = ROW[(code - 0x1E) as usize];
            Some(KeyCode::Char(if shift { c.to_ascii_uppercase() } else { c } as char))
        }
        0x27 => Some(KeyCode::Char(if shift { ':' } else { ';' })),
        0x28 => Some(KeyCode::Char(if shift { '"' } else { '\'' })),
        0x29 => Some(KeyCode::Char(if shift { '~' } else { '`' })),
        0x2B => Some(KeyCode::Char(if shift { '|' } else { '\\' })),
        0x2C..=0x32 => {
            const ROW: [u8; 7] = *b"zxcvbnm";
            let c = ROW[(code - 0x2C) as usize];
            Some(KeyCode::Char(if shift { c.to_ascii_uppercase() } else { c } as char))
        }
        0x33 => Some(KeyCode::Char(if shift { '<' } else { ',' })),
        0x34 => Some(KeyCode::Char(if shift { '>' } else { '.' })),
        0x35 => Some(KeyCode::Char(if shift { '?' } else { '/' })),
        0x39 => Some(KeyCode::Char(' ')),
        0x3B => Some(KeyCode::F1),
        0x3C => Some(KeyCode::F2),
        0x3D => Some(KeyCode::F3),
        0x3E => Some(KeyCode::F4),
        _ => None,
    }
}

pub fn poll() {
    unsafe {
        if !PRESENT {
            return;
        }
        for _ in 0..32 {
            let st = inb(STATUS_PORT);
            if (st & 0x01) == 0 {
                break;
            }
            let byte = inb(DATA_PORT);
            if (st & 0x20) != 0 {
                // Mouse byte
                if MOUSE_PHASE == 0 && (byte & 0x08) == 0 {
                    continue;
                }
                MOUSE_BYTES[MOUSE_PHASE as usize] = byte;
                MOUSE_PHASE += 1;
                if MOUSE_PHASE == 3 {
                    MOUSE_PHASE = 0;
                    let b0 = MOUSE_BYTES[0];
                    let dx = MOUSE_BYTES[1] as i8 as i32;
                    let dy = -(MOUSE_BYTES[2] as i8 as i32);
                    if dx != 0 || dy != 0 {
                        input::push(InputEvent::MouseMove { dx, dy });
                    }
                    let btns = b0 & 0x07;
                    let changed = btns ^ MOUSE_BUTTONS;
                    for bit in 0..3u8 {
                        if (changed & (1 << bit)) != 0 {
                            input::push(InputEvent::MouseButton {
                                button: bit,
                                pressed: (btns & (1 << bit)) != 0,
                            });
                        }
                    }
                    MOUSE_BUTTONS = btns;
                }
                continue;
            }

            if byte == 0xE0 {
                EXTENDED = true;
                continue;
            }
            let ext = EXTENDED;
            EXTENDED = false;

            let released = (byte & 0x80) != 0;
            let code = byte & 0x7F;
            match code {
                0x2A | 0x36 => {
                    SHIFT = !released;
                    continue;
                }
                0x1D => {
                    CTRL = !released;
                    continue;
                }
                0x38 => {
                    ALT = !released;
                    continue;
                }
                _ => {}
            }
            if let Some(key) = scancode_set1(code, SHIFT, ext) {
                if released {
                    input::push(InputEvent::KeyUp { key });
                } else {
                    input::push(InputEvent::KeyDown {
                        key,
                        shift: SHIFT,
                        ctrl: CTRL,
                        alt: ALT,
                    });
                }
            }
        }
    }
}
