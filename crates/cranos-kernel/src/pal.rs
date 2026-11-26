//! Bare-metal Platform Abstraction Layer (PAL) hooks for Rust `std`,
//! `cranos-web-time`, and standard C `libm` math symbols required by LLVM
//! codegen without any host libc or Linux runtime.

use core::alloc::{GlobalAlloc, Layout};
use core::arch::asm;

use crate::cpu;
use crate::kprintln;
use crate::serial;

#[unsafe(no_mangle)]
pub extern "C" fn cranos_clock_nanos() -> u64 {
    cpu::now_nanos()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_write(_fd: u32, write_buf: *const u8, nbytes: usize) {
    if write_buf.is_null() || nbytes == 0 {
        return;
    }
    let bytes = unsafe { core::slice::from_raw_parts(write_buf, nbytes) };
    if let Ok(s) = core::str::from_utf8(bytes) {
        serial::print_fmt(format_args!("{}", s));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_read(_fd: u32, _recv_buf: *mut u8, _nrequested: usize) -> usize {
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_log(msg_ptr: *const u8, len: usize) {
    unsafe { sys_write(2, msg_ptr, len) };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_panic(msg_ptr: *const u8, len: usize) -> ! {
    kprintln!("\n=== CRANOS PANIC ===");
    unsafe { sys_write(2, msg_ptr, len) };
    kprintln!("");
    loop {
        unsafe { asm!("cli; hlt") };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_halt() {
    loop {
        unsafe { asm!("cli; hlt") };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_rand(recv_buf: *mut u32, words: usize) {
    if recv_buf.is_null() {
        return;
    }
    let mut state = unsafe { core::arch::x86_64::_rdtsc() } ^ 0x9E37_79B9_7F4A_7C15;
    for i in 0..words {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        unsafe {
            recv_buf.add(i).write(state as u32);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_alloc_aligned(nbytes: usize, align: usize) -> *mut u8 {
    let align = align.max(8).next_power_of_two();
    let Ok(layout) = Layout::from_size_align(nbytes.max(1), align) else {
        return core::ptr::null_mut();
    };
    unsafe { crate::mem::ALLOCATOR.alloc(layout) }
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_getenv(
    _recv_buf: *mut u32,
    _words: usize,
    _varname: *const u8,
    _varname_len: usize,
) -> usize {
    usize::MAX
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_argc() -> usize {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_argv(_out_words: *mut u32, _out_nwords: usize, _arg_index: usize) -> usize {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_cycle_count() -> usize {
    unsafe { core::arch::x86_64::_rdtsc() as usize }
}

// ---------------------------------------------------------------------------
// C libm math ABI symbols backed by pure-Rust `libm` (no libc needed!)
// ---------------------------------------------------------------------------

macro_rules! export_math_f32_1 {
    ($($c_name:ident => $libm_fn:ident),* $(,)?) => {
        $(
            #[unsafe(no_mangle)]
            pub extern "C" fn $c_name(x: f32) -> f32 {
                libm::$libm_fn(x)
            }
        )*
    };
}

macro_rules! export_math_f32_2 {
    ($($c_name:ident => $libm_fn:ident),* $(,)?) => {
        $(
            #[unsafe(no_mangle)]
            pub extern "C" fn $c_name(x: f32, y: f32) -> f32 {
                libm::$libm_fn(x, y)
            }
        )*
    };
}

macro_rules! export_math_f64_1 {
    ($($c_name:ident => $libm_fn:ident),* $(,)?) => {
        $(
            #[unsafe(no_mangle)]
            pub extern "C" fn $c_name(x: f64) -> f64 {
                libm::$libm_fn(x)
            }
        )*
    };
}

macro_rules! export_math_f64_2 {
    ($($c_name:ident => $libm_fn:ident),* $(,)?) => {
        $(
            #[unsafe(no_mangle)]
            pub extern "C" fn $c_name(x: f64, y: f64) -> f64 {
                libm::$libm_fn(x, y)
            }
        )*
    };
}

export_math_f32_1!(
    sqrtf => sqrtf,
    floorf => floorf,
    ceilf => ceilf,
    roundf => roundf,
    roundevenf => rintf,
    rintf => rintf,
    truncf => truncf,
    fabsf => fabsf,
    sinf => sinf,
    cosf => cosf,
    tanf => tanf,
    asinf => asinf,
    acosf => acosf,
    atanf => atanf,
    sinhf => sinhf,
    coshf => coshf,
    tanhf => tanhf,
    expf => expf,
    exp2f => exp2f,
    expm1f => expm1f,
    logf => logf,
    log2f => log2f,
    log10f => log10f,
    log1pf => log1pf,
    cbrtf => cbrtf,
);

export_math_f32_2!(
    powf => powf,
    atan2f => atan2f,
    fmodf => fmodf,
    hypotf => hypotf,
    copysignf => copysignf,
    fminf => fminf,
    fmaxf => fmaxf,
);

export_math_f64_1!(
    sqrt => sqrt,
    floor => floor,
    ceil => ceil,
    round => round,
    roundeven => rint,
    rint => rint,
    trunc => trunc,
    fabs => fabs,
    sin => sin,
    cos => cos,
    tan => tan,
    asin => asin,
    acos => acos,
    atan => atan,
    sinh => sinh,
    cosh => cosh,
    tanh => tanh,
    exp => exp,
    exp2 => exp2,
    expm1 => expm1,
    log => log,
    log2 => log2,
    log10 => log10,
    log1p => log1p,
    cbrt => cbrt,
);

export_math_f64_2!(
    pow => pow,
    atan2 => atan2,
    fmod => fmod,
    hypot => hypot,
    copysign => copysign,
    fmin => fmin,
    fmax => fmax,
);

#[unsafe(no_mangle)]
pub extern "C" fn fmaf(a: f32, b: f32, c: f32) -> f32 {
    libm::fmaf(a, b, c)
}

#[unsafe(no_mangle)]
pub extern "C" fn fma(a: f64, b: f64, c: f64) -> f64 {
    libm::fma(a, b, c)
}

#[unsafe(no_mangle)]
pub extern "C" fn ldexpf(x: f32, n: i32) -> f32 {
    libm::ldexpf(x, n)
}

#[unsafe(no_mangle)]
pub extern "C" fn ldexp(x: f64, n: i32) -> f64 {
    libm::ldexp(x, n)
}
