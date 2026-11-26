use alloc::string::String;
use alloc::vec::Vec;

use cranos_llm::{InferenceState, Model, ParallelRunner, Sampler, Utf8StreamDecoder};

use crate::cpu;
use crate::kprintln;
use crate::smp;

pub struct SmpRunner {
    workers: usize,
}

impl SmpRunner {
    pub fn new(workers: usize) -> Self {
        Self {
            workers: workers.max(1),
        }
    }
}

impl ParallelRunner for SmpRunner {
    fn worker_count(&self) -> usize {
        self.workers
    }

    fn run_parallel(&self, work: &(dyn Fn(usize, usize) + Sync)) {
        smp::parallel_run(self.workers, &|wid, total| work(wid, total));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionPhase {
    Idle,
    Prefilling { done: usize, total: usize },
    Generating,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PerfMetrics {
    pub last_ttft_ms: u64,
    pub last_tok_per_sec_x10: u64, // fixed-point tenths (e.g. 495 = 49.5 tok/s)
    pub last_gb_per_sec_x10: u64,  // fixed-point tenths
    pub tokens_generated: usize,
    pub prompt_tokens: usize,
    pub total_tokens_ever: usize,
}

pub struct LlmSession {
    model: Model<'static>,
    state: InferenceState,
    runner: SmpRunner,
    sampler: Sampler,
    decoder: Utf8StreamDecoder,
    phase: SessionPhase,
    pending_prompt: Vec<u32>,
    prompt_cursor: usize,
    generated_tokens: Vec<u32>,
    max_gen_tokens: usize,
    next_token: u32,
    t_start_ns: u64,
    t_decode_start_ns: u64,
    pub metrics: PerfMetrics,
    pub parse_ms: u64,
}

impl LlmSession {
    /// Parses a GGUF buffer already resident in 1 GiB huge-page physical RAM.
    ///
    /// # Safety
    /// `phys_addr..phys_addr+size` must remain mapped and immutable for `'static`.
    pub unsafe fn load_from_phys(phys_addr: u64, size: u64, max_seq_len: usize) -> Result<Self, &'static str> {
        let t0 = cpu::now_nanos();
        let bytes: &'static [u8] =
            unsafe { core::slice::from_raw_parts(phys_addr as *const u8, size as usize) };
        let model = Model::load(bytes, max_seq_len)?;
        let state = InferenceState::new(&model.config);
        let parse_ms = cpu::now_nanos().saturating_sub(t0) / 1_000_000;

        // Use physical P-cores for memory-bandwidth-bound LLM matmul (matches
        // i5-12400's 6 physical P-cores without hyperthread thrashing).
        let workers = smp::physical_cores().min(smp::online_cores()).max(1);
        kprintln!(
            "[llm] Loaded {} ({}) in {}ms: {}L dim={} ffn={} heads={}/{} head_dim={} bytes/tok={}MiB workers={}",
            model.config.name,
            model.config.arch,
            parse_ms,
            model.config.n_layers,
            model.config.embed_dim,
            model.config.ffn_dim,
            model.config.n_heads,
            model.config.n_kv_heads,
            model.config.head_dim,
            model.bytes_per_token() / (1024 * 1024),
            workers
        );

        Ok(Self {
            model,
            state,
            runner: SmpRunner::new(workers),
            sampler: Sampler::chat(cpu::now_nanos()),
            decoder: Utf8StreamDecoder::default(),
            phase: SessionPhase::Idle,
            pending_prompt: Vec::new(),
            prompt_cursor: 0,
            generated_tokens: Vec::new(),
            max_gen_tokens: 256,
            next_token: 0,
            t_start_ns: 0,
            t_decode_start_ns: 0,
            metrics: PerfMetrics::default(),
            parse_ms,
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model.config.name
    }

    pub fn model_arch(&self) -> &str {
        &self.model.config.arch
    }

    pub fn model_size_mib(&self) -> u64 {
        self.model.config.total_weight_bytes / (1024 * 1024)
    }

    pub fn worker_count(&self) -> usize {
        self.runner.worker_count()
    }

    pub fn phase(&self) -> SessionPhase {
        self.phase
    }

    pub fn is_busy(&self) -> bool {
        self.phase != SessionPhase::Idle
    }

    pub fn stop(&mut self) {
        self.phase = SessionPhase::Idle;
    }

    pub fn clear_context(&mut self) {
        self.phase = SessionPhase::Idle;
        self.state.reset();
        self.pending_prompt.clear();
        self.generated_tokens.clear();
        self.decoder.clear();
    }

    pub fn start_chat(
        &mut self,
        turns: &[(bool, &str)],
        enable_thinking: bool,
        greedy: bool,
        max_tokens: usize,
    ) {
        self.state.reset();
        self.decoder.clear();
        self.generated_tokens.clear();
        self.sampler = if greedy {
            Sampler::greedy()
        } else {
            Sampler::chat(cpu::now_nanos())
        };
        let system = "You are CranOS, a bare-metal Rust AI operating system running directly on x86_64 hardware with Cranpose UI and zero Linux overhead. Be clear, technical, and concise.";
        let tokens = self
            .model
            .tokenizer
            .encode_chat_turns(system, turns, enable_thinking);

        self.metrics.prompt_tokens = tokens.len();
        self.metrics.tokens_generated = 0;
        self.pending_prompt = tokens;
        self.prompt_cursor = 0;
        self.max_gen_tokens = max_tokens;
        self.t_start_ns = cpu::now_nanos();
        self.t_decode_start_ns = 0;
        self.phase = if self.pending_prompt.is_empty() {
            SessionPhase::Idle
        } else {
            SessionPhase::Prefilling {
                done: 0,
                total: self.pending_prompt.len(),
            }
        };
    }

    /// Advances inference by one cooperative step (a small prefill slice or one
    /// decoded token) so Core 0 keeps input polling and Cranpose UI rendering
    /// live and responsive at all times.
    ///
    /// Returns `Some(text_piece)` when new text was produced, and sets
    /// `finished = true` when generation ends.
    pub fn step(&mut self) -> (Option<String>, bool) {
        match self.phase {
            SessionPhase::Idle => (None, false),
            SessionPhase::Prefilling { total, .. } => {
                // Process up to 4 prefill tokens per UI slice.
                let slice_end = (self.prompt_cursor + 4).min(total);
                for idx in self.prompt_cursor..slice_end {
                    let is_last = idx + 1 == total;
                    let tok = self.pending_prompt[idx];
                    if self
                        .model
                        .forward_token(tok, &mut self.state, &self.runner, is_last)
                        .is_err()
                    {
                        self.phase = SessionPhase::Idle;
                        return (Some(String::from("\n[context limit reached]")), true);
                    }
                }
                self.prompt_cursor = slice_end;
                if self.prompt_cursor >= total {
                    let now = cpu::now_nanos();
                    self.metrics.last_ttft_ms = now.saturating_sub(self.t_start_ns) / 1_000_000;
                    self.t_decode_start_ns = now;
                    self.next_token = self.sampler.sample(&mut self.state.logits, &self.generated_tokens);
                    self.phase = SessionPhase::Generating;
                } else {
                    self.phase = SessionPhase::Prefilling {
                        done: self.prompt_cursor,
                        total,
                    };
                }
                (None, false)
            }
            SessionPhase::Generating => {
                let tok = self.next_token;
                if self.model.tokenizer.is_stop_token(tok)
                    || self.generated_tokens.len() >= self.max_gen_tokens
                {
                    self.phase = SessionPhase::Idle;
                    self.update_live_metrics();
                    kprintln!(
                        "[llm] Done: {} tok | TTFT={}ms | {}.{} tok/s | {}.{} GB/s",
                        self.metrics.tokens_generated,
                        self.metrics.last_ttft_ms,
                        self.metrics.last_tok_per_sec_x10 / 10,
                        self.metrics.last_tok_per_sec_x10 % 10,
                        self.metrics.last_gb_per_sec_x10 / 10,
                        self.metrics.last_gb_per_sec_x10 % 10,
                    );
                    return (None, true);
                }

                self.generated_tokens.push(tok);
                self.metrics.tokens_generated = self.generated_tokens.len();
                self.metrics.total_tokens_ever += 1;

                let bytes = self.model.tokenizer.decode_token_bytes(tok);
                let piece = self.decoder.push_bytes(&bytes);

                if self
                    .model
                    .forward_token(tok, &mut self.state, &self.runner, true)
                    .is_err()
                {
                    self.phase = SessionPhase::Idle;
                    self.update_live_metrics();
                    return (Some(piece), true);
                }
                self.next_token = self
                    .sampler
                    .sample(&mut self.state.logits, &self.generated_tokens);
                self.update_live_metrics();

                (if piece.is_empty() { None } else { Some(piece) }, false)
            }
        }
    }

    fn update_live_metrics(&mut self) {
        let elapsed_ns = cpu::now_nanos().saturating_sub(self.t_decode_start_ns);
        let n = self.generated_tokens.len() as u128;
        if elapsed_ns > 0 && n > 0 {
            let tok_x10 = (n * 10_000_000_000u128) / (elapsed_ns as u128);
            self.metrics.last_tok_per_sec_x10 = tok_x10 as u64;
            let bytes = n * (self.model.bytes_per_token() as u128);
            let gb_x10 = (bytes * 10u128) / (elapsed_ns as u128);
            self.metrics.last_gb_per_sec_x10 = gb_x10 as u64;
        }
    }
}
