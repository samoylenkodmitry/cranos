use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use cranos_llm::{InferenceState, Model, ParallelRunner, Sampler, Utf8StreamDecoder};

struct SpinPoolShared {
    epoch: AtomicU64,
    done: AtomicU32,
    shutdown: AtomicBool,
    fn_ptr: AtomicUsize,
    ctx_ptr: AtomicUsize,
    workers: usize,
}

struct SpinPoolRunner {
    shared: Arc<SpinPoolShared>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl SpinPoolRunner {
    fn new(workers: usize) -> Self {
        let shared = Arc::new(SpinPoolShared {
            epoch: AtomicU64::new(0),
            done: AtomicU32::new(0),
            shutdown: AtomicBool::new(false),
            fn_ptr: AtomicUsize::new(0),
            ctx_ptr: AtomicUsize::new(0),
            workers: workers.max(1),
        });
        let mut handles = Vec::new();
        for wid in 1..workers {
            let s = Arc::clone(&shared);
            handles.push(std::thread::spawn(move || {
                let mut last_epoch = 0u64;
                loop {
                    let cur = s.epoch.load(Ordering::Acquire);
                    if cur == last_epoch {
                        if s.shutdown.load(Ordering::Relaxed) {
                            break;
                        }
                        std::hint::spin_loop();
                        continue;
                    }
                    last_epoch = cur;
                    let f: fn(usize, usize, usize) =
                        unsafe { core::mem::transmute(s.fn_ptr.load(Ordering::Relaxed)) };
                    let ctx = s.ctx_ptr.load(Ordering::Relaxed);
                    f(wid, s.workers, ctx);
                    s.done.fetch_add(1, Ordering::Release);
                }
            }));
        }
        Self { shared, handles }
    }
}

impl Drop for SpinPoolRunner {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

impl ParallelRunner for SpinPoolRunner {
    fn worker_count(&self) -> usize {
        self.shared.workers
    }

    fn run_parallel(&self, work: &(dyn Fn(usize, usize) + Sync)) {
        if self.shared.workers <= 1 {
            work(0, 1);
            return;
        }
        fn trampoline(wid: usize, total: usize, ctx: usize) {
            let w: &&(dyn Fn(usize, usize) + Sync) = unsafe { &*(ctx as *const &(dyn Fn(usize, usize) + Sync)) };
            w(wid, total);
        }
        let boxed_ref = &work;
        self.shared.done.store(0, Ordering::Relaxed);
        self.shared
            .fn_ptr
            .store(trampoline as *const () as usize, Ordering::Relaxed);
        self.shared
            .ctx_ptr
            .store(boxed_ref as *const _ as usize, Ordering::Relaxed);
        self.shared.epoch.fetch_add(1, Ordering::Release);
        work(0, self.shared.workers);
        let expected = (self.shared.workers - 1) as u32;
        while self.shared.done.load(Ordering::Acquire) < expected {
            std::hint::spin_loop();
        }
    }
}

fn run_model_smoke(path: &str, threads: usize, max_tokens: usize) {
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping {}: file not present", path);
        return;
    }
    let t_read = Instant::now();
    let bytes = std::fs::read(path).expect("read gguf");
    let read_ms = t_read.elapsed().as_millis();

    let t_load = Instant::now();
    let model = Model::load(&bytes, 2048).expect("load model");
    let load_ms = t_load.elapsed().as_millis();

    println!("\n=== {} ({}) ===", model.config.name, model.config.arch);
    println!(
        "layers={} embed={} ffn={} heads={}/{} head_dim={} vocab={} neox={}",
        model.config.n_layers,
        model.config.embed_dim,
        model.config.ffn_dim,
        model.config.n_heads,
        model.config.n_kv_heads,
        model.config.head_dim,
        model.config.vocab_size,
        model.config.rope_neox
    );
    println!(
        "SIMD: {:?} | read={}ms parse={}ms | bytes/tok={} MiB",
        model.caps,
        read_ms,
        load_ms,
        model.bytes_per_token() / (1024 * 1024)
    );

    let mut state = InferenceState::new(&model.config);
    let runner = SpinPoolRunner::new(threads);

    let prompt_tokens = model.tokenizer.encode_chat_turns(
        "You are CranOS, a helpful bare-metal Rust AI assistant. Answer concisely.",
        &[(true, "What is 2 + 2, and why is bare-metal OS programming in Rust cool?")],
        false,
    );
    println!("Prompt tokens ({}): {:?}", prompt_tokens.len(), &prompt_tokens);

    let t_prefill = Instant::now();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let is_last = i + 1 == prompt_tokens.len();
        model
            .forward_token(tok, &mut state, &runner, is_last)
            .expect("prefill");
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

    let mut sampler = Sampler::greedy();
    let mut decoder = Utf8StreamDecoder::default();
    let mut generated = Vec::new();
    let mut text = String::new();

    let mut next_tok = sampler.sample(&mut state.logits, &generated);
    let t_decode = Instant::now();
    for _ in 0..max_tokens {
        if model.tokenizer.is_stop_token(next_tok) {
            break;
        }
        generated.push(next_tok);
        let piece = decoder.push_bytes(&model.tokenizer.decode_token_bytes(next_tok));
        text.push_str(&piece);
        model
            .forward_token(next_tok, &mut state, &runner, true)
            .expect("decode");
        next_tok = sampler.sample(&mut state.logits, &generated);
    }
    let decode_secs = t_decode.elapsed().as_secs_f64();
    let tok_s = (generated.len() as f64) / decode_secs.max(1e-6);
    let bw_gbs = (generated.len() as f64 * model.bytes_per_token() as f64) / (decode_secs * 1e9);

    println!("Output:\n{}\n", text);
    println!(
        "Prefill: {} tok in {:.1} ms ({:.1} tok/s) | Decode: {} tok in {:.2} s = {:.2} tok/s ({:.1} GB/s, {} threads)",
        prompt_tokens.len(),
        prefill_ms,
        prompt_tokens.len() as f64 / (prefill_ms / 1000.0),
        generated.len(),
        decode_secs,
        tok_s,
        bw_gbs,
        threads
    );
    assert!(generated.len() >= 5, "expected coherent generated response");
    assert!(
        text.contains('4'),
        "expected answer to contain '4', got: {}",
        text
    );
}

#[test]
fn qwen3_0_6b_inference_and_bench() {
    run_model_smoke("/media/data/cranos/Qwen3-0.6B-Q8_0.gguf", 6, 48);
}

#[test]
fn qwen3_4b_inference_and_bench() {
    run_model_smoke("/media/data/cranos/Qwen3-4B-Q4_K_M.gguf", 6, 32);
}
