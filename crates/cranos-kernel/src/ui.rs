use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

use cranpose::{prelude::*, text::TextUnit};
use cranpose_core::MutableState;

use crate::llm_session::{PerfMetrics, SessionPhase};

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub is_user: bool,
    pub text: String,
    pub meta: String,
}

#[derive(Clone, Debug)]
pub struct SystemSummary {
    pub cpu_brand: String,
    pub logical_cores: usize,
    pub physical_cores: usize,
    pub ram_mib: usize,
    pub simd_label: String,
    pub huge_pages_1g: bool,
    pub barrier_ns: u64,
    pub gop_res: String,
    pub model_name: String,
    pub model_detail: String,
    pub has_model: bool,
    pub boot_ms: u64,
}

#[derive(Clone, Debug)]
pub enum UiAction {
    SendPrompt(String),
    StopGeneration,
    ToggleThinking,
    RunBenchmark,
    ClearChat,
}

pub struct UiShared {
    pub sys: SystemSummary,
    pub messages: Vec<ChatMessage>,
    pub draft: String,
    pub phase: SessionPhase,
    pub thinking: bool,
    pub perf: PerfMetrics,
    pub caret_visible: bool,
    pub actions: Vec<UiAction>,
    pub rev_state: Option<MutableState<u64>>,
    pub rev_counter: u64,
}

impl UiShared {
    pub fn new(sys: SystemSummary) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            sys,
            messages: Vec::new(),
            draft: String::new(),
            phase: SessionPhase::Idle,
            thinking: false,
            perf: PerfMetrics::default(),
            caret_visible: true,
            actions: Vec::new(),
            rev_state: None,
            rev_counter: 0,
        }))
    }

    pub fn notify(&mut self) {
        self.rev_counter = self.rev_counter.wrapping_add(1);
        if let Some(state) = self.rev_state {
            state.set(self.rev_counter);
        }
    }
}

fn txt(color: Color, size: f32) -> TextStyle {
    TextStyle {
        span_style: SpanStyle {
            color: Some(color),
            font_size: TextUnit::Sp(size),
            ..SpanStyle::default()
        },
        ..TextStyle::default()
    }
}

fn fmt_tenths(x10: u64) -> String {
    format!("{}.{}", x10 / 10, x10 % 10)
}

#[composable]
fn Pill(label: String, bg: Color, fg: Color) {
    Box(
        Modifier::empty()
            .background(bg)
            .rounded_corners(6.0)
            .padding_symmetric(10.0, 4.0),
        BoxSpec::default(),
        move || {
            Text(label.clone(), Modifier::empty(), txt(fg, 12.0));
        },
    );
}

#[composable]
fn ActionBtn(label: String, bg: Color, fg: Color, on_click: impl FnMut() + 'static) {
    Button(
        Modifier::empty()
            .background(bg)
            .rounded_corners(8.0)
            .padding_symmetric(14.0, 9.0),
        ButtonSpec::default(),
        on_click,
        move || {
            Text(label.clone(), Modifier::empty(), txt(fg, 13.0));
        },
    );
}

#[derive(Clone)]
pub struct UiHandle(pub Rc<RefCell<UiShared>>);

impl PartialEq for UiHandle {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

#[composable]
pub fn CranosApp(handle: UiHandle) {
    let shared = handle.0;
    let rev = cranpose_core::rememberMutableStateOf(|| 0u64);
    {
        let mut s = shared.borrow_mut();
        if s.rev_state.is_none() {
            s.rev_state = Some(rev);
        }
    }
    // Subscribe this root composition to external updates from the kernel loop.
    let _tick = rev.get();

    let (sys, messages, draft, phase, thinking, perf, caret) = {
        let s = shared.borrow();
        (
            s.sys.clone(),
            s.messages.clone(),
            s.draft.clone(),
            s.phase,
            s.thinking,
            s.perf,
            s.caret_visible,
        )
    };

    let bg_root = Color::from_rgb_u8(10, 14, 23);
    let bg_bar = Color::from_rgb_u8(15, 21, 34);
    let bg_card = Color::from_rgb_u8(17, 24, 39);
    let bg_user = Color::from_rgb_u8(28, 39, 64);
    let border_subtle = Color::from_rgb_u8(34, 46, 70);
    let accent_cyan = Color::from_rgb_u8(56, 189, 248);
    let accent_green = Color::from_rgb_u8(34, 197, 94);
    let accent_amber = Color::from_rgb_u8(251, 191, 36);
    let accent_rose = Color::from_rgb_u8(244, 63, 94);
    let text_primary = Color::from_rgb_u8(241, 245, 249);
    let text_muted = Color::from_rgb_u8(148, 163, 184);

    let (status_label, status_bg, status_fg) = if !sys.has_model {
        (
            String::from("NO MODEL"),
            Color::from_rgb_u8(69, 18, 30),
            accent_rose,
        )
    } else {
        match phase {
            SessionPhase::Idle => (
                String::from("READY"),
                Color::from_rgb_u8(16, 56, 36),
                accent_green,
            ),
            SessionPhase::Prefilling { done, total } => (
                format!("PREFILL {}/{}", done, total),
                Color::from_rgb_u8(68, 48, 12),
                accent_amber,
            ),
            SessionPhase::Generating => (
                String::from("STREAMING"),
                Color::from_rgb_u8(12, 54, 74),
                accent_cyan,
            ),
        }
    };

    Column(
        Modifier::empty().fill_max_size().background(bg_root),
        ColumnSpec::default(),
        move || {
            // 1. Top Bare-Metal OS Telemetry Header
            Row(
                Modifier::empty()
                    .fill_max_width()
                    .background(bg_bar)
                    .padding_symmetric(20.0, 12.0),
                RowSpec::default()
                    .horizontal_arrangement(LinearArrangement::SpaceBetween)
                    .vertical_alignment(VerticalAlignment::CenterVertically),
                {
                    let status_label = status_label.clone();
                    let sys = sys.clone();
                    move || {
                        Row(
                            Modifier::empty(),
                            RowSpec::default()
                                .horizontal_arrangement(LinearArrangement::spaced_by(12.0))
                                .vertical_alignment(VerticalAlignment::CenterVertically),
                            {
                                let status_label = status_label.clone();
                                let model_name = sys.model_name.clone();
                                move || {
                                    Text("CranOS", Modifier::empty(), txt(text_primary, 20.0));
                                    Pill(status_label.clone(), status_bg, status_fg);
                                    Text(model_name.clone(), Modifier::empty(), txt(accent_cyan, 14.0));
                                }
                            },
                        );

                        Row(
                            Modifier::empty(),
                            RowSpec::default()
                                .horizontal_arrangement(LinearArrangement::spaced_by(8.0))
                                .vertical_alignment(VerticalAlignment::CenterVertically),
                            {
                                let simd = sys.simd_label.clone();
                                let cores = sys.physical_cores;
                                let huge = sys.huge_pages_1g;
                                move || {
                                    if perf.last_tok_per_sec_x10 > 0 {
                                        Pill(
                                            format!(
                                                "{} tok/s  |  {} GB/s  |  TTFT {}ms",
                                                fmt_tenths(perf.last_tok_per_sec_x10),
                                                fmt_tenths(perf.last_gb_per_sec_x10),
                                                perf.last_ttft_ms
                                            ),
                                            Color::from_rgb_u8(14, 44, 62),
                                            accent_cyan,
                                        );
                                    }
                                    Pill(
                                        format!("{} P-Cores", cores),
                                        Color::from_rgb_u8(24, 34, 52),
                                        text_primary,
                                    );
                                    Pill(
                                        simd.clone(),
                                        Color::from_rgb_u8(24, 34, 52),
                                        text_muted,
                                    );
                                    if huge {
                                        Pill(
                                            String::from("1GiB Pages"),
                                            Color::from_rgb_u8(24, 34, 52),
                                            accent_green,
                                        );
                                    }
                                }
                            },
                        );
                    }
                },
            );

            // Thin divider
            Box(
                Modifier::empty()
                    .fill_max_width()
                    .height(1.0)
                    .background(border_subtle),
                BoxSpec::default(),
                || {},
            );

            // 2. Main Chat Area (takes remaining height)
            let shared_body = Rc::clone(&shared);
            let sys_body = sys.clone();
            let msgs_body = messages.clone();
            Column(
                Modifier::empty()
                    .fill_max_width()
                    .weight(1.0)
                    .padding_symmetric(28.0, 18.0),
                ColumnSpec::default().vertical_arrangement(LinearArrangement::spaced_by(14.0)),
                move || {
                    if msgs_body.is_empty() {
                        // Welcome + Hardware Diagnostics Card
                        Column(
                            Modifier::empty()
                                .fill_max_width()
                                .background(bg_card)
                                .rounded_corners(12.0)
                                .border(1.0, border_subtle, RoundedCornerShape::uniform(12.0))
                                .padding(22.0),
                            ColumnSpec::default()
                                .vertical_arrangement(LinearArrangement::spaced_by(10.0)),
                            {
                                let sys_c = sys_body.clone();
                                move || {
                                    Text(
                                        "Bare-Metal Rust AI Operating System (Zero Linux)",
                                        Modifier::empty(),
                                        txt(text_primary, 22.0),
                                    );
                                    Text(
                                        format!(
                                            "CPU: {}  |  Cores: {} logical / {} physical  |  SMP barrier: {} ns",
                                            sys_c.cpu_brand,
                                            sys_c.logical_cores,
                                            sys_c.physical_cores,
                                            sys_c.barrier_ns
                                        ),
                                        Modifier::empty(),
                                        txt(text_muted, 14.0),
                                    );
                                    Text(
                                        format!(
                                            "RAM: {} MiB usable  |  Display: {} (PAT Write-Combining)  |  Boot: {} ms",
                                            sys_c.ram_mib, sys_c.gop_res, sys_c.boot_ms
                                        ),
                                        Modifier::empty(),
                                        txt(text_muted, 14.0),
                                    );
                                    Text(
                                        format!("LLM: {}", sys_c.model_detail),
                                        Modifier::empty(),
                                        txt(accent_cyan, 14.0),
                                    );
                                }
                            },
                        );

                        // Quick Prompt Buttons
                        Text(
                            "Quick prompts (click or type below and press Enter):",
                            Modifier::empty(),
                            txt(text_muted, 13.0),
                        );
                        Row(
                            Modifier::empty().fill_max_width(),
                            RowSpec::default()
                                .horizontal_arrangement(LinearArrangement::spaced_by(10.0)),
                            {
                                let s1 = Rc::clone(&shared_body);
                                let s2 = Rc::clone(&shared_body);
                                let s3 = Rc::clone(&shared_body);
                                move || {
                                    let p1 = String::from(
                                        "Explain why running an LLM directly on bare metal without Linux reduces tail latency.",
                                    );
                                    let p2 = String::from(
                                        "Write a concise Rust function that implements binary search on a sorted slice.",
                                    );
                                    let s1c = Rc::clone(&s1);
                                    let s2c = Rc::clone(&s2);
                                    let s3c = Rc::clone(&s3);
                                    ActionBtn(
                                        String::from("Why bare-metal LLM?"),
                                        Color::from_rgb_u8(22, 34, 54),
                                        text_primary,
                                        move || {
                                            s1c.borrow_mut()
                                                .actions
                                                .push(UiAction::SendPrompt(p1.clone()));
                                        },
                                    );
                                    ActionBtn(
                                        String::from("Rust binary search"),
                                        Color::from_rgb_u8(22, 34, 54),
                                        text_primary,
                                        move || {
                                            s2c.borrow_mut()
                                                .actions
                                                .push(UiAction::SendPrompt(p2.clone()));
                                        },
                                    );
                                    ActionBtn(
                                        String::from("Run Speed Benchmark (F2)"),
                                        Color::from_rgb_u8(14, 50, 72),
                                        accent_cyan,
                                        move || {
                                            s3c.borrow_mut().actions.push(UiAction::RunBenchmark);
                                        },
                                    );
                                }
                            },
                        );
                    } else {
                        // Show the most recent messages so the latest output stays clearly in view.
                        let start = msgs_body.len().saturating_sub(6);
                        let visible = &msgs_body[start..];
                        for (idx, msg) in visible.iter().enumerate() {
                            let is_last = idx + 1 == visible.len();
                            let card_bg = if msg.is_user { bg_user } else { bg_card };
                            let role = if msg.is_user { "YOU" } else { "CRANOS" };
                            let role_color = if msg.is_user { accent_amber } else { accent_cyan };
                            let mut body = msg.text.clone();
                            if !msg.is_user && is_last && phase == SessionPhase::Generating {
                                body.push_str(" |");
                            } else if !msg.is_user && is_last && matches!(phase, SessionPhase::Prefilling { .. }) && body.is_empty() {
                                body.push_str("Prefilling prompt across bare-metal P-cores...");
                            }
                            let meta = msg.meta.clone();

                            Column(
                                Modifier::empty()
                                    .fill_max_width()
                                    .background(card_bg)
                                    .rounded_corners(10.0)
                                    .border(1.0, border_subtle, RoundedCornerShape::uniform(10.0))
                                    .padding_symmetric(16.0, 12.0),
                                ColumnSpec::default()
                                    .vertical_arrangement(LinearArrangement::spaced_by(6.0)),
                                move || {
                                    Row(
                                        Modifier::empty().fill_max_width(),
                                        RowSpec::default()
                                            .horizontal_arrangement(LinearArrangement::SpaceBetween),
                                        {
                                            let meta = meta.clone();
                                            move || {
                                                Text(role, Modifier::empty(), txt(role_color, 12.0));
                                                if !meta.is_empty() {
                                                    Text(meta.clone(), Modifier::empty(), txt(text_muted, 12.0));
                                                }
                                            }
                                        },
                                    );
                                    Text(body.clone(), Modifier::empty(), txt(text_primary, 15.0));
                                },
                            );
                        }
                    }
                },
            );

            // Thin divider above composer
            Box(
                Modifier::empty()
                    .fill_max_width()
                    .height(1.0)
                    .background(border_subtle),
                BoxSpec::default(),
                || {},
            );

            // 3. Bottom Input Composer + Controls
            let shared_bar = Rc::clone(&shared);
            let draft_bar = draft.clone();
            Row(
                Modifier::empty()
                    .fill_max_width()
                    .background(bg_bar)
                    .padding_symmetric(20.0, 14.0),
                RowSpec::default()
                    .horizontal_arrangement(LinearArrangement::spaced_by(10.0))
                    .vertical_alignment(VerticalAlignment::CenterVertically),
                move || {
                    let display_input = if draft_bar.is_empty() {
                        if caret {
                            String::from("| Type a prompt and press Enter... (F1 Think | F2 Bench | F3 Clear)")
                        } else {
                            String::from("  Type a prompt and press Enter... (F1 Think | F2 Bench | F3 Clear)")
                        }
                    } else if caret {
                        format!("{}|", draft_bar)
                    } else {
                        draft_bar.clone()
                    };
                    let input_color = if draft_bar.is_empty() {
                        text_muted
                    } else {
                        text_primary
                    };

                    Box(
                        Modifier::empty()
                            .weight(1.0)
                            .background(Color::from_rgb_u8(11, 16, 28))
                            .rounded_corners(8.0)
                            .border(1.0, border_subtle, RoundedCornerShape::uniform(8.0))
                            .padding_symmetric(14.0, 10.0),
                        BoxSpec::default(),
                        move || {
                            Text(display_input.clone(), Modifier::empty(), txt(input_color, 15.0));
                        },
                    );

                    let s_send = Rc::clone(&shared_bar);
                    let s_think = Rc::clone(&shared_bar);
                    let s_bench = Rc::clone(&shared_bar);
                    let s_clear = Rc::clone(&shared_bar);

                    if phase != SessionPhase::Idle {
                        ActionBtn(
                            String::from("Stop (Esc)"),
                            Color::from_rgb_u8(120, 24, 44),
                            text_primary,
                            move || {
                                s_send.borrow_mut().actions.push(UiAction::StopGeneration);
                            },
                        );
                    } else {
                        let d_send = draft_bar.clone();
                        ActionBtn(
                            String::from("Send"),
                            Color::from_rgb_u8(14, 116, 144),
                            text_primary,
                            move || {
                                if !d_send.trim().is_empty() {
                                    s_send
                                        .borrow_mut()
                                        .actions
                                        .push(UiAction::SendPrompt(d_send.clone()));
                                }
                            },
                        );
                    }

                    ActionBtn(
                        if thinking {
                            String::from("Think: ON")
                        } else {
                            String::from("Think: OFF")
                        },
                        if thinking {
                            Color::from_rgb_u8(56, 42, 110)
                        } else {
                            Color::from_rgb_u8(24, 34, 52)
                        },
                        text_primary,
                        move || {
                            s_think.borrow_mut().actions.push(UiAction::ToggleThinking);
                        },
                    );

                    ActionBtn(
                        String::from("Bench"),
                        Color::from_rgb_u8(24, 34, 52),
                        accent_cyan,
                        move || {
                            s_bench.borrow_mut().actions.push(UiAction::RunBenchmark);
                        },
                    );

                    ActionBtn(
                        String::from("Clear"),
                        Color::from_rgb_u8(24, 34, 52),
                        text_muted,
                        move || {
                            s_clear.borrow_mut().actions.push(UiAction::ClearChat);
                        },
                    );
                },
            );
        },
    );
}
