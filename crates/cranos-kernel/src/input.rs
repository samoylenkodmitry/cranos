use core::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyCode {
    Char(char),
    Enter,
    Backspace,
    Delete,
    Tab,
    Escape,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    PageUp,
    PageDown,
    F1,
    F2,
    F3,
    F4,
    F10,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputEvent {
    KeyDown {
        key: KeyCode,
        shift: bool,
        ctrl: bool,
        alt: bool,
    },
    KeyUp {
        key: KeyCode,
    },
    MouseMove {
        dx: i32,
        dy: i32,
    },
    MouseAbs {
        x: u16, // 0..=32767
        y: u16, // 0..=32767
    },
    MouseButton {
        button: u8, // 0=left, 1=right, 2=middle
        pressed: bool,
    },
    MouseWheel {
        delta: i8,
    },
}

const QUEUE_CAP: usize = 256;

struct RingQueue {
    buf: [Option<InputEvent>; QUEUE_CAP],
    head: usize,
    tail: usize,
}

impl RingQueue {
    const fn new() -> Self {
        Self {
            buf: [None; QUEUE_CAP],
            head: 0,
            tail: 0,
        }
    }

    fn push(&mut self, ev: InputEvent) {
        let next = (self.tail + 1) % QUEUE_CAP;
        if next == self.head {
            self.head = (self.head + 1) % QUEUE_CAP;
        }
        self.buf[self.tail] = Some(ev);
        self.tail = next;
    }

    fn pop(&mut self) -> Option<InputEvent> {
        if self.head == self.tail {
            return None;
        }
        let ev = self.buf[self.head].take();
        self.head = (self.head + 1) % QUEUE_CAP;
        ev
    }
}

static LOCK: AtomicBool = AtomicBool::new(false);
static mut QUEUE: RingQueue = RingQueue::new();

fn lock() {
    while LOCK
        .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        core::hint::spin_loop();
    }
}

fn unlock() {
    LOCK.store(false, Ordering::Release);
}

pub fn push(ev: InputEvent) {
    lock();
    unsafe {
        (*(&raw mut QUEUE)).push(ev);
    }
    unlock();
}

pub fn pop() -> Option<InputEvent> {
    lock();
    let ev = unsafe { (*(&raw mut QUEUE)).pop() };
    unlock();
    ev
}

/// Converts a USB HID Usage ID (Keyboard/Keypad page 0x07) + modifier byte into a KeyCode.
pub fn hid_usage_to_keycode(usage: u8, shift: bool) -> Option<KeyCode> {
    match usage {
        0x04..=0x1D => {
            let base = b'a' + (usage - 0x04);
            let ch = if shift { base.to_ascii_uppercase() } else { base } as char;
            Some(KeyCode::Char(ch))
        }
        0x1E..=0x27 => {
            const UNSHIFTED: [u8; 10] = *b"1234567890";
            const SHIFTED: [u8; 10] = *b"!@#$%^&*()";
            let idx = (usage - 0x1E) as usize;
            let b = if shift { SHIFTED[idx] } else { UNSHIFTED[idx] };
            Some(KeyCode::Char(b as char))
        }
        0x28 => Some(KeyCode::Enter),
        0x29 => Some(KeyCode::Escape),
        0x2A => Some(KeyCode::Backspace),
        0x2B => Some(KeyCode::Tab),
        0x2C => Some(KeyCode::Char(' ')),
        0x2D => Some(KeyCode::Char(if shift { '_' } else { '-' })),
        0x2E => Some(KeyCode::Char(if shift { '+' } else { '=' })),
        0x2F => Some(KeyCode::Char(if shift { '{' } else { '[' })),
        0x30 => Some(KeyCode::Char(if shift { '}' } else { ']' })),
        0x31 => Some(KeyCode::Char(if shift { '|' } else { '\\' })),
        0x33 => Some(KeyCode::Char(if shift { ':' } else { ';' })),
        0x34 => Some(KeyCode::Char(if shift { '"' } else { '\'' })),
        0x35 => Some(KeyCode::Char(if shift { '~' } else { '`' })),
        0x36 => Some(KeyCode::Char(if shift { '<' } else { ',' })),
        0x37 => Some(KeyCode::Char(if shift { '>' } else { '.' })),
        0x38 => Some(KeyCode::Char(if shift { '?' } else { '/' })),
        0x3A => Some(KeyCode::F1),
        0x3B => Some(KeyCode::F2),
        0x3C => Some(KeyCode::F3),
        0x3D => Some(KeyCode::F4),
        0x43 => Some(KeyCode::F10),
        0x4A => Some(KeyCode::Home),
        0x4B => Some(KeyCode::PageUp),
        0x4C => Some(KeyCode::Delete),
        0x4D => Some(KeyCode::End),
        0x4E => Some(KeyCode::PageDown),
        0x4F => Some(KeyCode::Right),
        0x50 => Some(KeyCode::Left),
        0x51 => Some(KeyCode::Down),
        0x52 => Some(KeyCode::Up),
        _ => None,
    }
}

/// Polls COM1 serial input so headless QEMU testing or serial terminals can type into CranOS.
pub fn poll_serial() {
    for _ in 0..32 {
        let Some(b) = crate::serial::try_read_byte() else {
            break;
        };
        let key = match b {
            b'\r' | b'\n' => Some(KeyCode::Enter),
            0x08 | 0x7F => Some(KeyCode::Backspace),
            b'\t' => Some(KeyCode::Tab),
            0x1B => Some(KeyCode::Escape),
            0x20..=0x7E => Some(KeyCode::Char(b as char)),
            _ => None,
        };
        if let Some(key) = key {
            push(InputEvent::KeyDown {
                key,
                shift: false,
                ctrl: false,
                alt: false,
            });
            push(InputEvent::KeyUp { key });
        }
    }
}
