use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::gguf::{GgmlType, GgufFile};
use crate::math::{f32_exp, f32_sqrt, f64_pow, f64_sin_cos, silu};
use crate::quant::{
    QuantInput, SimdCaps, axpy_f16_vec, dequantize_row, dot_f16_vec, f32_to_f16_slice,
    matvec_row_range, needs_q8_0, needs_q8_k,
};
use crate::tokenizer::Tokenizer;

/// Abstraction over bare-metal SMP spin-barrier pool (`cranos-kernel`) and
/// host `std::thread::scope` (Linux verification & benchmarks).
pub trait ParallelRunner {
    fn worker_count(&self) -> usize;
    fn run_parallel(&self, work: &(dyn Fn(usize, usize) + Sync));
}

pub struct SerialRunner;

impl ParallelRunner for SerialRunner {
    fn worker_count(&self) -> usize {
        1
    }

    fn run_parallel(&self, work: &(dyn Fn(usize, usize) + Sync)) {
        work(0, 1);
    }
}

#[derive(Clone, Copy, Debug)]
struct WeightMatrix<'a> {
    ty: GgmlType,
    cols: usize,
    rows: usize,
    data: &'a [u8],
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub arch: String,
    pub name: String,
    pub n_layers: usize,
    pub embed_dim: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_freq_base: f32,
    pub rope_neox: bool,
    pub total_weight_bytes: u64,
}

struct LayerWeights<'a> {
    attn_norm: Vec<f32>,
    attn_q: WeightMatrix<'a>,
    attn_k: WeightMatrix<'a>,
    attn_v: WeightMatrix<'a>,
    attn_output: WeightMatrix<'a>,
    attn_q_bias: Option<Vec<f32>>,
    attn_k_bias: Option<Vec<f32>>,
    attn_v_bias: Option<Vec<f32>>,
    attn_q_norm: Option<Vec<f32>>,
    attn_k_norm: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    ffn_gate: WeightMatrix<'a>,
    ffn_up: WeightMatrix<'a>,
    ffn_down: WeightMatrix<'a>,
}

pub struct Model<'a> {
    pub config: ModelConfig,
    pub tokenizer: Tokenizer,
    pub caps: SimdCaps,
    token_embd: WeightMatrix<'a>,
    output_norm: Vec<f32>,
    output: WeightMatrix<'a>,
    layers: Vec<LayerWeights<'a>>,
    rope_cos: Vec<f32>, // [max_seq_len * (head_dim / 2)]
    rope_sin: Vec<f32>,
}

pub struct KvCache {
    pub max_seq_len: usize,
    kv_dim: usize,
    // F16 KV cache: 2x lower bandwidth than F32, hardware F16C conversion
    k_cache: Vec<u16>,
    v_cache: Vec<u16>,
    pub pos: usize,
}

impl KvCache {
    pub fn new(cfg: &ModelConfig) -> Self {
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let total = cfg.n_layers * cfg.max_seq_len * kv_dim;
        Self {
            max_seq_len: cfg.max_seq_len,
            kv_dim,
            k_cache: vec![0u16; total],
            v_cache: vec![0u16; total],
            pos: 0,
        }
    }

    pub fn clear(&mut self) {
        self.pos = 0;
    }

    #[inline]
    fn offset(&self, layer: usize, pos: usize) -> usize {
        (layer * self.max_seq_len + pos) * self.kv_dim
    }
}

/// Pre-allocated per-session inference workspace (zero heap allocations during token decode).
pub struct InferenceState {
    pub kv: KvCache,
    x: Vec<f32>,
    xb: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    proj_out: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    pub logits: Vec<f32>,
    scores: Vec<f32>,
    quant_in: QuantInput,
}

impl InferenceState {
    pub fn new(cfg: &ModelConfig) -> Self {
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let max_dim = cfg.embed_dim.max(cfg.ffn_dim).max(q_dim);
        Self {
            kv: KvCache::new(cfg),
            x: vec![0.0; cfg.embed_dim],
            xb: vec![0.0; cfg.embed_dim],
            q: vec![0.0; q_dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            attn_out: vec![0.0; q_dim],
            proj_out: vec![0.0; cfg.embed_dim],
            gate: vec![0.0; cfg.ffn_dim],
            up: vec![0.0; cfg.ffn_dim],
            logits: vec![0.0; cfg.vocab_size],
            scores: vec![0.0; cfg.n_heads * cfg.max_seq_len],
            quant_in: QuantInput::with_max_dim(max_dim),
        }
    }

    pub fn reset(&mut self) {
        self.kv.clear();
    }
}

fn load_matrix<'a>(gguf: &GgufFile<'a>, name: &str) -> Result<WeightMatrix<'a>, &'static str> {
    let (info, data) = gguf.tensor_bytes(name).ok_or("missing weight tensor")?;
    let cols = info.dims[0];
    let rows = if info.n_dims >= 2 { info.dims[1] } else { 1 };
    Ok(WeightMatrix {
        ty: info.ty,
        cols,
        rows,
        data,
    })
}

fn load_f32_vec(gguf: &GgufFile<'_>, name: &str) -> Result<Vec<f32>, &'static str> {
    let (info, data) = gguf.tensor_bytes(name).ok_or("missing 1D weight tensor")?;
    let len = info.dims[0];
    let mut out = vec![0.0f32; len];
    dequantize_row(info.ty, data, &mut out);
    Ok(out)
}

fn load_optional_f32_vec(gguf: &GgufFile<'_>, name: &str) -> Option<Vec<f32>> {
    let (info, data) = gguf.tensor_bytes(name)?;
    let len = info.dims[0];
    let mut out = vec![0.0f32; len];
    dequantize_row(info.ty, data, &mut out);
    Some(out)
}

impl<'a> Model<'a> {
    pub fn load(gguf_bytes: &'a [u8], max_seq_len: usize) -> Result<Self, &'static str> {
        let gguf = GgufFile::parse(gguf_bytes)?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;

        let arch = String::from(gguf.get_str("general.architecture").unwrap_or("qwen3"));
        let name = String::from(gguf.get_str("general.name").unwrap_or("Qwen3"));

        let get_arch_u64 = |suffix: &str| -> Option<u64> {
            let key = format!("{}.{}", arch, suffix);
            gguf.get_u64(&key)
        };
        let get_arch_f32 = |suffix: &str| -> Option<f32> {
            let key = format!("{}.{}", arch, suffix);
            gguf.get_f32(&key)
        };

        let n_layers = get_arch_u64("block_count").ok_or("missing block_count")? as usize;
        let embed_dim = get_arch_u64("embedding_length").ok_or("missing embedding_length")? as usize;
        let ffn_dim = get_arch_u64("feed_forward_length").ok_or("missing feed_forward_length")? as usize;
        let n_heads = get_arch_u64("attention.head_count").ok_or("missing head_count")? as usize;
        let n_kv_heads = get_arch_u64("attention.head_count_kv").unwrap_or(n_heads as u64) as usize;

        let token_embd = load_matrix(&gguf, "token_embd.weight")?;
        let vocab_size = token_embd.rows;
        let output_norm = load_f32_vec(&gguf, "output_norm.weight")?;
        // Qwen3-0.6B/1.7B/4B tie lm_head to token_embd when output.weight is absent.
        let tied_output = gguf.tensor_bytes("output.weight").is_none();
        let output = load_matrix(&gguf, "output.weight").unwrap_or(token_embd);

        // Derive head_dim from metadata or blk.0.attn_q.weight rows (CRITICAL for Qwen3!).
        let q0 = load_matrix(&gguf, "blk.0.attn_q.weight")?;
        let head_dim = get_arch_u64("attention.key_length")
            .map(|v| v as usize)
            .unwrap_or_else(|| q0.rows / n_heads);

        let rms_norm_eps = get_arch_f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let rope_freq_base = get_arch_f32("rope.freq_base").unwrap_or(1_000_000.0);
        // Qwen2 / Qwen3 use NeoX split-half RoPE; Llama uses interleaved RoPE.
        let rope_neox = arch.starts_with("qwen") || arch.starts_with("gemma") || arch.starts_with("phi");

        let ctx_len = get_arch_u64("context_length")
            .map(|v| (v as usize).min(max_seq_len))
            .unwrap_or(max_seq_len);

        let mut layers = Vec::with_capacity(n_layers);
        let mut total_weight_bytes = if tied_output {
            token_embd.data.len() as u64
        } else {
            (token_embd.data.len() + output.data.len()) as u64
        };
        for l in 0..n_layers {
            let attn_norm = load_f32_vec(&gguf, &format!("blk.{l}.attn_norm.weight"))?;
            let attn_q = load_matrix(&gguf, &format!("blk.{l}.attn_q.weight"))?;
            let attn_k = load_matrix(&gguf, &format!("blk.{l}.attn_k.weight"))?;
            let attn_v = load_matrix(&gguf, &format!("blk.{l}.attn_v.weight"))?;
            let attn_output = load_matrix(&gguf, &format!("blk.{l}.attn_output.weight"))?;
            let attn_q_bias = load_optional_f32_vec(&gguf, &format!("blk.{l}.attn_q.bias"));
            let attn_k_bias = load_optional_f32_vec(&gguf, &format!("blk.{l}.attn_k.bias"));
            let attn_v_bias = load_optional_f32_vec(&gguf, &format!("blk.{l}.attn_v.bias"));
            let attn_q_norm = load_optional_f32_vec(&gguf, &format!("blk.{l}.attn_q_norm.weight"));
            let attn_k_norm = load_optional_f32_vec(&gguf, &format!("blk.{l}.attn_k_norm.weight"));
            let ffn_norm = load_f32_vec(&gguf, &format!("blk.{l}.ffn_norm.weight"))?;
            let ffn_gate = load_matrix(&gguf, &format!("blk.{l}.ffn_gate.weight"))?;
            let ffn_up = load_matrix(&gguf, &format!("blk.{l}.ffn_up.weight"))?;
            let ffn_down = load_matrix(&gguf, &format!("blk.{l}.ffn_down.weight"))?;

            total_weight_bytes += (attn_q.data.len()
                + attn_k.data.len()
                + attn_v.data.len()
                + attn_output.data.len()
                + ffn_gate.data.len()
                + ffn_up.data.len()
                + ffn_down.data.len()) as u64;

            layers.push(LayerWeights {
                attn_norm,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                attn_q_bias,
                attn_k_bias,
                attn_v_bias,
                attn_q_norm,
                attn_k_norm,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        // Precompute exact RoPE cos/sin table for positions 0..ctx_len.
        let half = head_dim / 2;
        let mut rope_cos = vec![0.0f32; ctx_len * half];
        let mut rope_sin = vec![0.0f32; ctx_len * half];
        let mut inv_freq = vec![0.0f64; half];
        for i in 0..half {
            inv_freq[i] = 1.0 / f64_pow(rope_freq_base as f64, (2 * i) as f64 / (head_dim as f64));
        }
        for pos in 0..ctx_len {
            let p = pos as f64;
            let row = pos * half;
            for i in 0..half {
                let (s, c) = f64_sin_cos(p * inv_freq[i]);
                rope_sin[row + i] = s as f32;
                rope_cos[row + i] = c as f32;
            }
        }

        Ok(Self {
            config: ModelConfig {
                arch,
                name,
                n_layers,
                embed_dim,
                ffn_dim,
                n_heads,
                n_kv_heads,
                head_dim,
                vocab_size,
                max_seq_len: ctx_len,
                rms_norm_eps,
                rope_freq_base,
                rope_neox,
                total_weight_bytes,
            },
            tokenizer,
            caps: SimdCaps::detect(),
            token_embd,
            output_norm,
            output,
            layers,
            rope_cos,
            rope_sin,
        })
    }

    /// Bytes read from weight matrices per generated token (for memory bandwidth telemetry).
    pub fn bytes_per_token(&self) -> u64 {
        let mut bytes = self.output.data.len() as u64;
        for l in &self.layers {
            bytes += (l.attn_q.data.len()
                + l.attn_k.data.len()
                + l.attn_v.data.len()
                + l.attn_output.data.len()
                + l.ffn_gate.data.len()
                + l.ffn_up.data.len()
                + l.ffn_down.data.len()) as u64;
        }
        bytes
    }

    fn prepare_input(x: &[f32], q_in: &mut QuantInput, types: &[GgmlType]) {
        let mut want_q8_0 = false;
        let mut want_q8_k = false;
        for &ty in types {
            if needs_q8_0(ty) {
                want_q8_0 = true;
            }
            if needs_q8_k(ty) {
                want_q8_k = true;
            }
        }
        if want_q8_0 {
            q_in.prepare_q8_0(x);
        }
        if want_q8_k {
            q_in.prepare_q8_k(x);
        }
    }

    /// Parallel matrix-vector multiply: `out = W * x` across all P-cores.
    fn matvec_parallel(
        &self,
        w: WeightMatrix<'_>,
        x: &[f32],
        q_in: &QuantInput,
        out: &mut [f32],
        runner: &dyn ParallelRunner,
    ) {
        let rows = w.rows;
        let cols = w.cols;
        let workers = runner.worker_count().max(1);
        if workers == 1 || rows < 64 {
            matvec_row_range(w.ty, w.data, cols, 0, rows, x, q_in, out, self.caps);
            return;
        }
        // Align chunk boundaries to multiples of 4 rows so the 4-row unrolled SIMD
        // kernel runs cleanly on every worker.
        let out_addr = out.as_mut_ptr() as usize;
        let caps = self.caps;
        runner.run_parallel(&|wid, total| {
            let chunk_raw = (rows + total - 1) / total;
            let chunk = ((chunk_raw + 3) & !3).max(4);
            let start = (wid * chunk).min(rows);
            let end = (start + chunk).min(rows);
            if start < end {
                let dst = unsafe {
                    core::slice::from_raw_parts_mut((out_addr as *mut f32).add(start), end - start)
                };
                matvec_row_range(w.ty, w.data, cols, start, end, x, q_in, dst, caps);
            }
        });
    }

    /// Fused parallel dispatch for Q, K, V projections sharing a single barrier!
    fn matvec_qkv_parallel(
        &self,
        wq: WeightMatrix<'_>,
        wk: WeightMatrix<'_>,
        wv: WeightMatrix<'_>,
        x: &[f32],
        q_in: &QuantInput,
        q_out: &mut [f32],
        k_out: &mut [f32],
        v_out: &mut [f32],
        runner: &dyn ParallelRunner,
    ) {
        let total_rows = wq.rows + wk.rows + wv.rows;
        let rq = wq.rows;
        let rqk = wq.rows + wk.rows;
        let q_addr = q_out.as_mut_ptr() as usize;
        let k_addr = k_out.as_mut_ptr() as usize;
        let v_addr = v_out.as_mut_ptr() as usize;
        let caps = self.caps;

        runner.run_parallel(&|wid, total| {
            let chunk = ((total_rows + total - 1) / total + 3) & !3;
            let s = (wid * chunk).min(total_rows);
            let e = (s + chunk).min(total_rows);
            if s >= e {
                return;
            }
            // Slice of Q
            if s < rq {
                let qs = s;
                let qe = e.min(rq);
                let dst = unsafe {
                    core::slice::from_raw_parts_mut((q_addr as *mut f32).add(qs), qe - qs)
                };
                matvec_row_range(wq.ty, wq.data, wq.cols, qs, qe, x, q_in, dst, caps);
            }
            // Slice of K
            if e > rq && s < rqk {
                let ks = s.saturating_sub(rq);
                let ke = e.min(rqk) - rq;
                let dst = unsafe {
                    core::slice::from_raw_parts_mut((k_addr as *mut f32).add(ks), ke - ks)
                };
                matvec_row_range(wk.ty, wk.data, wk.cols, ks, ke, x, q_in, dst, caps);
            }
            // Slice of V
            if e > rqk {
                let vs = s.saturating_sub(rqk);
                let ve = e - rqk;
                let dst = unsafe {
                    core::slice::from_raw_parts_mut((v_addr as *mut f32).add(vs), ve - vs)
                };
                matvec_row_range(wv.ty, wv.data, wv.cols, vs, ve, x, q_in, dst, caps);
            }
        });
    }

    /// Fused parallel dispatch for FFN gate + up projections sharing a single barrier!
    fn matvec_pair_parallel(
        &self,
        wa: WeightMatrix<'_>,
        wb: WeightMatrix<'_>,
        x: &[f32],
        q_in: &QuantInput,
        a_out: &mut [f32],
        b_out: &mut [f32],
        runner: &dyn ParallelRunner,
    ) {
        let ra = wa.rows;
        let total_rows = wa.rows + wb.rows;
        let a_addr = a_out.as_mut_ptr() as usize;
        let b_addr = b_out.as_mut_ptr() as usize;
        let caps = self.caps;

        runner.run_parallel(&|wid, total| {
            let chunk = ((total_rows + total - 1) / total + 3) & !3;
            let s = (wid * chunk).min(total_rows);
            let e = (s + chunk).min(total_rows);
            if s >= e {
                return;
            }
            if s < ra {
                let sa = s;
                let ea = e.min(ra);
                let dst = unsafe {
                    core::slice::from_raw_parts_mut((a_addr as *mut f32).add(sa), ea - sa)
                };
                matvec_row_range(wa.ty, wa.data, wa.cols, sa, ea, x, q_in, dst, caps);
            }
            if e > ra {
                let sb = s.saturating_sub(ra);
                let eb = e - ra;
                let dst = unsafe {
                    core::slice::from_raw_parts_mut((b_addr as *mut f32).add(sb), eb - sb)
                };
                matvec_row_range(wb.ty, wb.data, wb.cols, sb, eb, x, q_in, dst, caps);
            }
        });
    }

    /// Runs one token through all transformer layers and optionally computes LM head logits.
    /// Pass `compute_logits = false` for intermediate prefill tokens to skip the large
    /// 151,936-row vocab projection!
    pub fn forward_token(
        &self,
        token: u32,
        state: &mut InferenceState,
        runner: &dyn ParallelRunner,
        compute_logits: bool,
    ) -> Result<(), &'static str> {
        let pos = state.kv.pos;
        if pos >= self.config.max_seq_len {
            return Err("KV cache context window full");
        }
        let tok = (token as usize).min(self.config.vocab_size - 1);

        // 1. Lookup and dequantize token embedding into state.x.
        let emb_stride = self.token_embd.ty.row_bytes(self.token_embd.cols);
        let emb_row = &self.token_embd.data[tok * emb_stride..(tok + 1) * emb_stride];
        dequantize_row(self.token_embd.ty, emb_row, &mut state.x);

        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let half = head_dim / 2;
        let kv_group = n_heads / n_kv_heads;
        let attn_scale = 1.0 / f32_sqrt(head_dim as f32);
        let eps = self.config.rms_norm_eps;
        let rope_c = &self.rope_cos[pos * half..(pos + 1) * half];
        let rope_s = &self.rope_sin[pos * half..(pos + 1) * half];
        let rope_neox = self.config.rope_neox;

        // 2. Transformer layers
        for (l, layer) in self.layers.iter().enumerate() {
            // Pre-attention RMSNorm
            rms_norm(&state.x, &layer.attn_norm, eps, &mut state.xb);

            // Quantize normalized hidden state once for fused Q/K/V matvec
            Self::prepare_input(
                &state.xb,
                &mut state.quant_in,
                &[layer.attn_q.ty, layer.attn_k.ty, layer.attn_v.ty],
            );
            self.matvec_qkv_parallel(
                layer.attn_q,
                layer.attn_k,
                layer.attn_v,
                &state.xb,
                &state.quant_in,
                &mut state.q,
                &mut state.k,
                &mut state.v,
                runner,
            );

            // Optional QKV biases (Qwen2/2.5)
            if let Some(bq) = &layer.attn_q_bias {
                add_in_place(&mut state.q, bq);
            }
            if let Some(bk) = &layer.attn_k_bias {
                add_in_place(&mut state.k, bk);
            }
            if let Some(bv) = &layer.attn_v_bias {
                add_in_place(&mut state.v, bv);
            }

            // Per-head QK-Norm (Qwen3!)
            if let Some(q_norm) = &layer.attn_q_norm {
                for h in 0..n_heads {
                    let s = h * head_dim;
                    rms_norm_in_place(&mut state.q[s..s + head_dim], q_norm, eps);
                }
            }
            if let Some(k_norm) = &layer.attn_k_norm {
                for h in 0..n_kv_heads {
                    let s = h * head_dim;
                    rms_norm_in_place(&mut state.k[s..s + head_dim], k_norm, eps);
                }
            }

            // Rotary Position Embeddings (RoPE) on Q and K
            for h in 0..n_heads {
                apply_rope(&mut state.q[h * head_dim..(h + 1) * head_dim], rope_c, rope_s, rope_neox);
            }
            for h in 0..n_kv_heads {
                apply_rope(&mut state.k[h * head_dim..(h + 1) * head_dim], rope_c, rope_s, rope_neox);
            }

            // Store K and V into F16 KV cache at (l, pos)
            let kv_off = state.kv.offset(l, pos);
            let kv_dim = state.kv.kv_dim;
            f32_to_f16_slice(
                &state.k,
                &mut state.kv.k_cache[kv_off..kv_off + kv_dim],
                self.caps,
            );
            f32_to_f16_slice(
                &state.v,
                &mut state.kv.v_cache[kv_off..kv_off + kv_dim],
                self.caps,
            );

            // Grouped-Query Attention (parallelized across heads on all P-cores)
            let seq_len = pos + 1;
            let layer_kv_base = l * state.kv.max_seq_len * kv_dim;
            let k_layer = &state.kv.k_cache[layer_kv_base..layer_kv_base + seq_len * kv_dim];
            let v_layer = &state.kv.v_cache[layer_kv_base..layer_kv_base + seq_len * kv_dim];
            let q_slice: &[f32] = &state.q;
            let attn_out_addr = state.attn_out.as_mut_ptr() as usize;
            let scores_addr = state.scores.as_mut_ptr() as usize;
            let max_seq = state.kv.max_seq_len;
            let caps = self.caps;

            let attn_head_work = |wid: usize, total: usize| {
                let heads_per = (n_heads + total - 1) / total;
                let h_start = (wid * heads_per).min(n_heads);
                let h_end = (h_start + heads_per).min(n_heads);
                for h in h_start..h_end {
                    let qh = &q_slice[h * head_dim..(h + 1) * head_dim];
                    let kv_h = h / kv_group;
                    let kv_h_off = kv_h * head_dim;
                    let scores = unsafe {
                        core::slice::from_raw_parts_mut(
                            (scores_addr as *mut f32).add(h * max_seq),
                            seq_len,
                        )
                    };

                    let mut max_score = f32::NEG_INFINITY;
                    for t in 0..seq_len {
                        let kh = &k_layer[t * kv_dim + kv_h_off..t * kv_dim + kv_h_off + head_dim];
                        let s = dot_f16_vec(kh, qh, caps) * attn_scale;
                        scores[t] = s;
                        if s > max_score {
                            max_score = s;
                        }
                    }

                    let mut sum_exp = 0.0f32;
                    for t in 0..seq_len {
                        let e = f32_exp(scores[t] - max_score);
                        scores[t] = e;
                        sum_exp += e;
                    }
                    let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

                    let out_h = unsafe {
                        core::slice::from_raw_parts_mut(
                            (attn_out_addr as *mut f32).add(h * head_dim),
                            head_dim,
                        )
                    };
                    out_h.fill(0.0);
                    for t in 0..seq_len {
                        let w = scores[t] * inv_sum;
                        let vh = &v_layer[t * kv_dim + kv_h_off..t * kv_dim + kv_h_off + head_dim];
                        axpy_f16_vec(out_h, vh, w, caps);
                    }
                }
            };

            if seq_len >= 16 && runner.worker_count() > 1 {
                runner.run_parallel(&attn_head_work);
            } else {
                attn_head_work(0, 1);
            }

            // Attention output projection + residual add
            Self::prepare_input(&state.attn_out, &mut state.quant_in, &[layer.attn_output.ty]);
            self.matvec_parallel(
                layer.attn_output,
                &state.attn_out,
                &state.quant_in,
                &mut state.proj_out,
                runner,
            );
            add_in_place(&mut state.x, &state.proj_out);

            // Pre-FFN RMSNorm
            rms_norm(&state.x, &layer.ffn_norm, eps, &mut state.xb);

            // Fused FFN gate + up projections
            Self::prepare_input(
                &state.xb,
                &mut state.quant_in,
                &[layer.ffn_gate.ty, layer.ffn_up.ty],
            );
            self.matvec_pair_parallel(
                layer.ffn_gate,
                layer.ffn_up,
                &state.xb,
                &state.quant_in,
                &mut state.gate,
                &mut state.up,
                runner,
            );

            // SwiGLU activation: gate[i] = silu(gate[i]) * up[i]
            for i in 0..self.config.ffn_dim {
                state.gate[i] = silu(state.gate[i]) * state.up[i];
            }

            // FFN down projection + residual add
            Self::prepare_input(&state.gate, &mut state.quant_in, &[layer.ffn_down.ty]);
            self.matvec_parallel(
                layer.ffn_down,
                &state.gate,
                &state.quant_in,
                &mut state.proj_out,
                runner,
            );
            add_in_place(&mut state.x, &state.proj_out);
        }

        state.kv.pos += 1;

        // 3. Final RMSNorm + LM Head logits (only when predicting next token)
        if compute_logits {
            rms_norm(&state.x, &self.output_norm, eps, &mut state.xb);
            Self::prepare_input(&state.xb, &mut state.quant_in, &[self.output.ty]);
            self.matvec_parallel(
                self.output,
                &state.xb,
                &state.quant_in,
                &mut state.logits,
                runner,
            );
        }

        Ok(())
    }
}

#[inline]
fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let scale = 1.0 / f32_sqrt(ss / (x.len() as f32) + eps);
    for i in 0..x.len() {
        out[i] = x[i] * scale * weight[i];
    }
}

#[inline]
fn rms_norm_in_place(x: &mut [f32], weight: &[f32], eps: f32) {
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let scale = 1.0 / f32_sqrt(ss / (x.len() as f32) + eps);
    for i in 0..x.len() {
        x[i] = x[i] * scale * weight[i];
    }
}

#[inline]
fn add_in_place(dst: &mut [f32], src: &[f32]) {
    for i in 0..dst.len() {
        dst[i] += src[i];
    }
}

#[inline]
fn apply_rope(head: &mut [f32], cos: &[f32], sin: &[f32], neox: bool) {
    let half = cos.len();
    if neox {
        for i in 0..half {
            let x0 = head[i];
            let x1 = head[i + half];
            let c = cos[i];
            let s = sin[i];
            head[i] = x0 * c - x1 * s;
            head[i + half] = x0 * s + x1 * c;
        }
    } else {
        for i in 0..half {
            let x0 = head[2 * i];
            let x1 = head[2 * i + 1];
            let c = cos[i];
            let s = sin[i];
            head[2 * i] = x0 * c - x1 * s;
            head[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    rng_state: u64,
}

impl Sampler {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repeat_penalty: 1.0,
            rng_state: 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub fn chat(seed: u64) -> Self {
        Self {
            temperature: 0.6,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.08,
            rng_state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        ((x >> 40) as f32) / ((1u32 << 24) as f32)
    }

    pub fn sample(&mut self, logits: &mut [f32], recent_tokens: &[u32]) -> u32 {
        if self.repeat_penalty > 1.0 {
            let start = recent_tokens.len().saturating_sub(64);
            for &tok in &recent_tokens[start..] {
                let idx = tok as usize;
                if idx < logits.len() {
                    if logits[idx] > 0.0 {
                        logits[idx] /= self.repeat_penalty;
                    } else {
                        logits[idx] *= self.repeat_penalty;
                    }
                }
            }
        }

        if self.temperature <= 0.01 || self.top_k <= 1 {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            return best_idx as u32;
        }

        // Top-K selection into a small sorted buffer (no full vocab sort!).
        let k = self.top_k.min(64).min(logits.len());
        let mut top: [(usize, f32); 64] = [(0, f32::NEG_INFINITY); 64];
        let mut min_top = f32::NEG_INFINITY;
        let mut min_pos = 0usize;

        for (i, &v) in logits.iter().enumerate() {
            if v > min_top {
                top[min_pos] = (i, v);
                min_top = top[0].1;
                min_pos = 0;
                for j in 1..k {
                    if top[j].1 < min_top {
                        min_top = top[j].1;
                        min_pos = j;
                    }
                }
            }
        }

        let cand = &mut top[..k];
        cand.sort_by(|a, b| b.1.total_cmp(&a.1));

        let max_v = cand[0].1;
        let inv_t = 1.0 / self.temperature;
        let mut probs = [0.0f32; 64];
        let mut sum = 0.0f32;
        for i in 0..k {
            let p = f32_exp((cand[i].1 - max_v) * inv_t);
            probs[i] = p;
            sum += p;
        }

        // Top-P truncation
        let mut cum = 0.0f32;
        let mut cut = k;
        for i in 0..k {
            probs[i] /= sum;
            cum += probs[i];
            if cum >= self.top_p {
                cut = i + 1;
                break;
            }
        }
        let mut norm = 0.0f32;
        for i in 0..cut {
            norm += probs[i];
        }
        let r = self.next_f32() * norm;
        let mut acc = 0.0f32;
        for i in 0..cut {
            acc += probs[i];
            if r <= acc {
                return cand[i].0 as u32;
            }
        }
        cand[0].0 as u32
    }
}
