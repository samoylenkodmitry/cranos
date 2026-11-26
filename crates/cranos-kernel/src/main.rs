#![no_main]

extern crate alloc;

mod acpi;
mod cpu;
mod fb;
mod gdt;
mod idt;
mod input;
mod llm_session;
mod mem;
mod pal;
mod ps2;
mod serial;
mod smp;
mod ui;
mod xhci;

use alloc::rc::Rc;
use core::arch::asm;
use std::sync::atomic::{AtomicU64, Ordering};

use cranos_boot_info::{BOOT_INFO_MAGIC, BootInfo};
use cranpose::framebuffer::{Framebuffer as CranposeFramebuffer, PixelRect};
use cranpose_foundation::PointerButton;
use cranpose_services::SystemTheme;

use crate::input::{InputEvent, KeyCode};
use crate::llm_session::{LlmSession, SessionPhase};
use crate::ui::{ChatMessage, CranosApp, SystemSummary, UiAction, UiHandle, UiShared};

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.boot")]
pub extern "C" fn _start(boot_info_ptr: u64) -> ! {
    kernel_main(boot_info_ptr)
}

fn render_cranpose_frame(
    ui: &mut CranposeFramebuffer,
    hw_fb: &mut fb::Framebuffer,
    dirty: &mut Vec<PixelRect>,
    rgba_scratch: &mut Vec<u8>,
    cursor_pos: (f32, f32),
    show_cursor: bool,
) {
    if ui.tick(cpu::now_nanos()) {
        ui.changed_areas(dirty);
        if !dirty.is_empty() {
            hw_fb.hide_cursor();
            for area in dirty.iter() {
                let need = area.width as usize * area.height as usize * 4;
                if rgba_scratch.len() < need {
                    rgba_scratch.resize(need, 0);
                }
                ui.draw_area(*area, &mut rgba_scratch[..need]);
                hw_fb.blit_rgba_rect(
                    area.x,
                    area.y,
                    area.width,
                    area.height,
                    &rgba_scratch[..need],
                );
            }
            if show_cursor {
                hw_fb.draw_cursor(cursor_pos.0 as u32, cursor_pos.1 as u32);
            }
        }
    }
}

fn start_prompt(
    shared: &Rc<std::cell::RefCell<UiShared>>,
    session: &mut Option<LlmSession>,
    prompt: String,
    greedy: bool,
    max_tokens: usize,
) {
    let prompt_trimmed = prompt.trim().to_string();
    if prompt_trimmed.is_empty() {
        return;
    }

    let thinking = {
        let mut s = shared.borrow_mut();
        s.draft.clear();
        s.messages.push(ChatMessage {
            is_user: true,
            text: prompt_trimmed.clone(),
            meta: String::new(),
        });
        if session.is_none() {
            s.messages.push(ChatMessage {
                is_user: false,
                text: String::from(
                    "No model.gguf was found at boot. Place a Qwen3 GGUF at /model.gguf on a FAT32 USB drive or /cranos/model.gguf on an ext4 NVMe partition.",
                ),
                meta: String::from("offline"),
            });
            s.notify();
            return;
        }
        s.messages.push(ChatMessage {
            is_user: false,
            text: String::new(),
            meta: String::from("starting..."),
        });
        s.notify();
        s.thinking
    };

    if let Some(sess) = session.as_mut() {
        let history: Vec<(bool, String)> = {
            let s = shared.borrow();
            // Exclude the empty assistant placeholder we just appended.
            let end = s.messages.len().saturating_sub(1);
            let start = end.saturating_sub(6);
            s.messages[start..end]
                .iter()
                .map(|m| (m.is_user, m.text.clone()))
                .collect()
        };
        let turn_refs: Vec<(bool, &str)> = history.iter().map(|(u, t)| (*u, t.as_str())).collect();
        kprintln!("[chat] User: {}", prompt_trimmed);
        sess.start_chat(&turn_refs, thinking, greedy, max_tokens);
        let mut s = shared.borrow_mut();
        s.phase = sess.phase();
        s.notify();
    }
}

fn kernel_main(boot_info_ptr: u64) -> ! {
    serial::init();
    kprintln!("\n==========================================");
    kprintln!("  CranOS v0.1 — Bare-Metal Rust LLM OS");
    kprintln!("==========================================");

    gdt::init();
    idt::init();

    let boot_info = unsafe { &*(boot_info_ptr as *const BootInfo) };
    if boot_info.magic != BOOT_INFO_MAGIC {
        kprintln!("FATAL: invalid BootInfo magic: {:#x}", boot_info.magic);
        loop {
            unsafe { asm!("cli; hlt") };
        }
    }

    cpu::enable_simd_on_current_core();
    cpu::configure_pat_wc();
    let feats = cpu::detect_features();
    let tsc_hz = cpu::calibrate_tsc();
    let mem_stats = mem::init(boot_info);

    std::panic::set_hook(std::boxed::Box::new(|info| {
        kprintln!("\n=== CRANOS RUST PANIC ===\n{}", info);
        loop {
            unsafe { asm!("cli; hlt") };
        }
    }));

    cpu::set_identity_range_cache_attr(
        boot_info.pml4_phys_addr,
        boot_info.fb.phys_addr,
        boot_info.fb.size,
        cpu::PAT_WC_BITS,
    );

    kprintln!("CPU: {}", feats.brand_str());
    kprintln!(
        "SIMD: AVX2={} FMA={} F16C={} AVX-VNNI={} 1GiB-Pages={} PAT={}",
        feats.avx2,
        feats.fma,
        feats.f16c,
        feats.avx_vnni,
        boot_info.has_1gib_pages != 0,
        feats.pat
    );
    kprintln!(
        "TSC: {} MHz | RAM: {} MiB | GOP: {}x{} @ {:#x}",
        tsc_hz / 1_000_000,
        mem_stats.total_bytes / (1024 * 1024),
        boot_info.fb.width,
        boot_info.fb.height,
        boot_info.fb.phys_addr
    );

    smp::init(boot_info.acpi_rsdp_addr, boot_info.pml4_phys_addr);

    let worker_mask = AtomicU64::new(0);
    let t0 = cpu::now_nanos();
    for _ in 0..200 {
        smp::parallel_run(smp::online_cores(), &|wid, _total| {
            worker_mask.fetch_or(1u64 << wid, Ordering::Relaxed);
        });
    }
    let barrier_ns = cpu::now_nanos().saturating_sub(t0) / 200;
    kprintln!(
        "[smp] 200 barriers across {} cores: {} ns/barrier (mask={:#x})",
        smp::online_cores(),
        barrier_ns,
        worker_mask.load(Ordering::Relaxed)
    );

    ps2::init();
    xhci::init(boot_info.pml4_phys_addr);

    let mut session: Option<LlmSession> = if boot_info.model_phys_addr != 0 && boot_info.model_size > 0 {
        kprintln!(
            "[llm] Parsing GGUF '{}' ({} MiB) @ {:#x}...",
            boot_info.model_name_str(),
            boot_info.model_size / (1024 * 1024),
            boot_info.model_phys_addr
        );
        match unsafe { LlmSession::load_from_phys(boot_info.model_phys_addr, boot_info.model_size, 2048) } {
            Ok(s) => Some(s),
            Err(err) => {
                kprintln!("[llm] Failed to parse GGUF: {}", err);
                None
            }
        }
    } else {
        kprintln!("[llm] No model.gguf loaded by UEFI stage");
        None
    };

    let simd_label = if feats.avx_vnni {
        String::from("AVX-VNNI + AVX2")
    } else if feats.avx2 && feats.fma {
        String::from("AVX2 + FMA")
    } else {
        String::from("SSE2")
    };

    let (model_name, model_detail, has_model) = match &session {
        Some(s) => (
            format!("{} ({} MiB)", s.model_name(), s.model_size_mib()),
            format!(
                "{} [{}]  |  {} MiB resident in 1GiB pages  |  parsed in {} ms  |  {} inference P-cores",
                s.model_name(),
                s.model_arch(),
                s.model_size_mib(),
                s.parse_ms,
                s.worker_count()
            ),
            true,
        ),
        None => (
            String::from("No Model Loaded"),
            String::from("Place model.gguf on FAT32 USB (/model.gguf) or ext4 NVMe (/cranos/model.gguf)"),
            false,
        ),
    };

    let boot_ms = cpu::now_nanos() / 1_000_000;
    let sys = SystemSummary {
        cpu_brand: String::from(feats.brand_str()),
        logical_cores: smp::online_cores(),
        physical_cores: smp::physical_cores(),
        ram_mib: mem_stats.total_bytes / (1024 * 1024),
        simd_label,
        huge_pages_1g: boot_info.has_1gib_pages != 0,
        barrier_ns,
        gop_res: format!("{}x{}", boot_info.fb.width, boot_info.fb.height),
        model_name,
        model_detail,
        has_model,
        boot_ms,
    };

    let shared = UiShared::new(sys);
    let shared_for_ui = Rc::clone(&shared);

    let mut hw_fb = fb::Framebuffer::new(boot_info.fb);
    let mut ui = CranposeFramebuffer::new(
        boot_info.fb.width,
        boot_info.fb.height,
        1.0,
        SystemTheme::Dark,
        &[],
        move || CranosApp(UiHandle(Rc::clone(&shared_for_ui))),
    )
    .expect("create Cranpose Framebuffer");
    ui.set_active(true);

    let mut dirty: Vec<PixelRect> = Vec::new();
    let mut rgba_scratch: Vec<u8> = Vec::new();
    let mut cursor_x = (boot_info.fb.width as f32) * 0.5;
    let mut cursor_y = (boot_info.fb.height as f32) * 0.5;
    let mut cursor_moved = false;

    render_cranpose_frame(
        &mut ui,
        &mut hw_fb,
        &mut dirty,
        &mut rgba_scratch,
        (cursor_x, cursor_y),
        cursor_moved,
    );
    kprintln!(
        "[CRANOS_READY] boot={}ms model_loaded={}",
        cpu::now_nanos() / 1_000_000,
        has_model
    );

    let mut last_blink_ns = cpu::now_nanos();
    let mut last_stream_render_ns = cpu::now_nanos();

    loop {
        ps2::poll();
        xhci::poll();
        input::poll_serial();

        // 1. Process hardware input events (USB xHCI, PS/2, COM1 serial)
        while let Some(ev) = input::pop() {
            match ev {
                InputEvent::MouseMove { dx, dy } => {
                    cursor_x = (cursor_x + dx as f32).clamp(0.0, (boot_info.fb.width - 1) as f32);
                    cursor_y = (cursor_y + dy as f32).clamp(0.0, (boot_info.fb.height - 1) as f32);
                    cursor_moved = true;
                    ui.mouse_move(cursor_x, cursor_y);
                    hw_fb.draw_cursor(cursor_x as u32, cursor_y as u32);
                }
                InputEvent::MouseButton { button, pressed } => {
                    cursor_moved = true;
                    let btn = match button {
                        0 => PointerButton::Primary,
                        1 => PointerButton::Secondary,
                        _ => continue,
                    };
                    ui.mouse_move(cursor_x, cursor_y);
                    if pressed {
                        ui.mouse_button_pressed(btn);
                    } else {
                        ui.mouse_button_released(btn);
                    }
                }
                InputEvent::MouseWheel { delta } => {
                    ui.mouse_scroll(0.0, (delta as f32) * 32.0);
                }
                InputEvent::KeyDown { key, ctrl, .. } => match key {
                    KeyCode::Char(c) if ctrl && (c == 'c' || c == 'C') => {
                        shared.borrow_mut().actions.push(UiAction::StopGeneration);
                    }
                    KeyCode::Char(c) if ctrl && (c == 'l' || c == 'L') => {
                        shared.borrow_mut().actions.push(UiAction::ClearChat);
                    }
                    KeyCode::Char(c) => {
                        let mut s = shared.borrow_mut();
                        if s.draft.len() < 512 {
                            s.draft.push(c);
                            s.caret_visible = true;
                            s.notify();
                        }
                    }
                    KeyCode::Backspace => {
                        let mut s = shared.borrow_mut();
                        if s.draft.pop().is_some() {
                            s.caret_visible = true;
                            s.notify();
                        }
                    }
                    KeyCode::Enter => {
                        let draft = shared.borrow().draft.clone();
                        if !draft.trim().is_empty() {
                            shared.borrow_mut().actions.push(UiAction::SendPrompt(draft));
                        }
                    }
                    KeyCode::Escape => {
                        shared.borrow_mut().actions.push(UiAction::StopGeneration);
                    }
                    KeyCode::F1 => {
                        shared.borrow_mut().actions.push(UiAction::ToggleThinking);
                    }
                    KeyCode::F2 => {
                        shared.borrow_mut().actions.push(UiAction::RunBenchmark);
                    }
                    KeyCode::F3 => {
                        shared.borrow_mut().actions.push(UiAction::ClearChat);
                    }
                    _ => {}
                },
                InputEvent::KeyUp { .. } => {}
            }
        }

        // 2. Drain UI actions queued by Cranpose button clicks or keyboard shortcuts
        let pending_actions: Vec<UiAction> = core::mem::take(&mut shared.borrow_mut().actions);
        for action in pending_actions {
            match action {
                UiAction::SendPrompt(text) => {
                    if !session.as_ref().is_some_and(LlmSession::is_busy) {
                        start_prompt(&shared, &mut session, text, false, 256);
                    }
                }
                UiAction::StopGeneration => {
                    if let Some(sess) = session.as_mut() {
                        sess.stop();
                        let mut s = shared.borrow_mut();
                        s.phase = SessionPhase::Idle;
                        s.notify();
                    }
                }
                UiAction::ToggleThinking => {
                    let mut s = shared.borrow_mut();
                    s.thinking = !s.thinking;
                    s.notify();
                }
                UiAction::RunBenchmark => {
                    if !session.as_ref().is_some_and(LlmSession::is_busy) {
                        start_prompt(
                            &shared,
                            &mut session,
                            String::from(
                                "What is 2 + 2, and why is bare-metal OS programming in Rust fast?",
                            ),
                            true,
                            48,
                        );
                    }
                }
                UiAction::ClearChat => {
                    if let Some(sess) = session.as_mut() {
                        sess.clear_context();
                    }
                    let mut s = shared.borrow_mut();
                    s.messages.clear();
                    s.phase = SessionPhase::Idle;
                    s.notify();
                }
            }
        }

        // 3. Cooperative bare-metal LLM inference step across all P-cores
        let mut force_render = false;
        if let Some(sess) = session.as_mut()
            && sess.is_busy()
        {
            let (piece, finished) = sess.step();
            let now = cpu::now_nanos();
            let mut s = shared.borrow_mut();
            s.phase = sess.phase();
            s.perf = sess.metrics;
            if let Some(text) = piece {
                kprint!("{}", text);
                if let Some(last) = s.messages.last_mut()
                    && !last.is_user
                {
                    last.text.push_str(&text);
                    last.meta = format!(
                        "{} tok  |  {}.{} tok/s  |  {}.{} GB/s",
                        sess.metrics.tokens_generated,
                        sess.metrics.last_tok_per_sec_x10 / 10,
                        sess.metrics.last_tok_per_sec_x10 % 10,
                        sess.metrics.last_gb_per_sec_x10 / 10,
                        sess.metrics.last_gb_per_sec_x10 % 10,
                    );
                }
            }
            if finished {
                kprintln!("");
                if let Some(last) = s.messages.last_mut()
                    && !last.is_user
                {
                    last.meta = format!(
                        "{} tok  |  {}.{} tok/s  |  {}.{} GB/s  |  TTFT {}ms",
                        sess.metrics.tokens_generated,
                        sess.metrics.last_tok_per_sec_x10 / 10,
                        sess.metrics.last_tok_per_sec_x10 % 10,
                        sess.metrics.last_gb_per_sec_x10 / 10,
                        sess.metrics.last_gb_per_sec_x10 % 10,
                        sess.metrics.last_ttft_ms
                    );
                }
                s.notify();
                force_render = true;
            } else if now.saturating_sub(last_stream_render_ns) >= 33_000_000 {
                // Cap streaming UI redraws to ~30 Hz so Cranpose CPU drawing
                // consumes < 2% of inference time on fast 50+ tok/s models!
                last_stream_render_ns = now;
                s.notify();
                force_render = true;
            }
        } else {
            // Caret blink every 550ms while idle
            let now = cpu::now_nanos();
            if now.saturating_sub(last_blink_ns) >= 550_000_000 {
                last_blink_ns = now;
                let mut s = shared.borrow_mut();
                s.caret_visible = !s.caret_visible;
                s.notify();
            }
        }

        // 4. Tick Cranpose and blit dirty rects to the PAT WC GOP framebuffer
        let busy_streaming = session.as_ref().is_some_and(LlmSession::is_busy);
        if !busy_streaming || force_render {
            render_cranpose_frame(
                &mut ui,
                &mut hw_fb,
                &mut dirty,
                &mut rgba_scratch,
                (cursor_x, cursor_y),
                cursor_moved,
            );
        }

        if !busy_streaming {
            smp::cpu_relax();
        }
    }
}
