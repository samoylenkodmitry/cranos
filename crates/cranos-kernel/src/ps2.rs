use crate::input::{self, InputEvent, KeyCode};
use crate::serial::inb;

const DATA_PORT: u16 = 0x60;
const STATUS_PORT: u16 = 0x64;

static mut PRESENT: bool = false;
static mut SHIFT: bool = false;
static mut CTRL: bool = false;
static mut ALT: bool = false;
static mut EXTENDED: bool = false;

pub fn init() {
    let status = unsafe { inb(STATUS_PORT) };
    // No i8042 controller on USB-only systems (returns floating bus 0xFF).
    if status == 0xFF {
        unsafe { PRESENT = false };
        return;
    }

    // Drain any stale bytes without probing the AUX/MUX port (Clevo/TUXEDO
    // laptop ECs such as samarch require i8042.noaux / i8042.nomux and can
    // lock up if AUX commands 0xA8/0xD4 are written).
    unsafe {
        PRESENT = true;
        for _ in 0..32 {
            if (inb(STATUS_PORT) & 0x01) == 0 {
                break;
            }
            let _ = inb(DATA_PORT);
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
        0x44 => Some(KeyCode::F10),
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
                // Ignore any stray AUX byte (mouse input comes over USB xHCI).
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
