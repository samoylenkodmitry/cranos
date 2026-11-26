use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::gguf::{GgufFile, GgufValue};

/// GPT-2 / Qwen byte-level BPE tokenizer loaded directly from GGUF metadata.
pub struct Tokenizer {
    vocab: Vec<String>,
    token_to_id: BTreeMap<String, u32>,
    merges: BTreeMap<(u32, u32), u32>,
    special_tokens: Vec<(String, u32)>,
    byte_to_char: [char; 256],
    char_to_byte: BTreeMap<char, u8>,
    pub bos_id: Option<u32>,
    pub eos_id: u32,
    pub im_start_id: Option<u32>,
    pub im_end_id: Option<u32>,
    pub think_start_id: Option<u32>,
    pub think_end_id: Option<u32>,
}

fn build_gpt2_byte_tables() -> ([char; 256], BTreeMap<char, u8>) {
    let mut b2c = ['\0'; 256];
    let mut c2b = BTreeMap::new();
    let mut is_direct = [false; 256];
    for b in 33..=126usize {
        is_direct[b] = true;
    }
    for b in 161..=172usize {
        is_direct[b] = true;
    }
    for b in 174..=255usize {
        is_direct[b] = true;
    }

    let mut next_cp = 256u32;
    for b in 0..256usize {
        let cp = if is_direct[b] {
            b as u32
        } else {
            let v = next_cp;
            next_cp += 1;
            v
        };
        let ch = char::from_u32(cp).unwrap_or('?');
        b2c[b] = ch;
        c2b.insert(ch, b as u8);
    }
    (b2c, c2b)
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Result<Self, &'static str> {
        let tokens_raw = match gguf.kv.get("tokenizer.ggml.tokens") {
            Some(GgufValue::StrArray(v)) => v,
            _ => return Err("missing tokenizer.ggml.tokens"),
        };

        let mut vocab = Vec::with_capacity(tokens_raw.len());
        let mut token_to_id = BTreeMap::new();
        for (i, &tok) in tokens_raw.iter().enumerate() {
            let s = String::from(tok);
            token_to_id.insert(s.clone(), i as u32);
            vocab.push(s);
        }

        let mut merges = BTreeMap::new();
        if let Some(GgufValue::StrArray(merges_raw)) = gguf.kv.get("tokenizer.ggml.merges") {
            for (rank, &line) in merges_raw.iter().enumerate() {
                if let Some((a, b)) = line.split_once(' ') {
                    if let (Some(&id_a), Some(&id_b)) = (token_to_id.get(a), token_to_id.get(b)) {
                        merges.insert((id_a, id_b), rank as u32);
                    }
                }
            }
        }

        // Collect special / control tokens (token_type == 3 in GGUF) plus known Qwen/Llama tags.
        let mut special_tokens = Vec::new();
        if let Some(GgufValue::I32Array(types)) = gguf.kv.get("tokenizer.ggml.token_type") {
            for (i, &ty) in types.iter().enumerate() {
                if ty == 3 && i < vocab.len() && !vocab[i].is_empty() {
                    special_tokens.push((vocab[i].clone(), i as u32));
                }
            }
        }
        for tag in [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<think>",
            "</think>",
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
        ] {
            if let Some(&id) = token_to_id.get(tag) {
                if !special_tokens.iter().any(|(_, eid)| *eid == id) {
                    special_tokens.push((String::from(tag), id));
                }
            }
        }
        // Longest-first matching for special tokens.
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let bos_id = gguf.get_u64("tokenizer.ggml.bos_token_id").map(|v| v as u32);
        let eos_id = gguf
            .get_u64("tokenizer.ggml.eos_token_id")
            .map(|v| v as u32)
            .or_else(|| token_to_id.get("<|im_end|>").copied())
            .or_else(|| token_to_id.get("<|endoftext|>").copied())
            .unwrap_or(0);

        let im_start_id = token_to_id.get("<|im_start|>").copied();
        let im_end_id = token_to_id.get("<|im_end|>").copied();
        let think_start_id = token_to_id.get("<think>").copied();
        let think_end_id = token_to_id.get("</think>").copied();

        let (byte_to_char, char_to_byte) = build_gpt2_byte_tables();

        Ok(Self {
            vocab,
            token_to_id,
            merges,
            special_tokens,
            byte_to_char,
            char_to_byte,
            bos_id,
            eos_id,
            im_start_id,
            im_end_id,
            think_start_id,
            think_end_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn is_stop_token(&self, id: u32) -> bool {
        id == self.eos_id || self.im_end_id == Some(id)
    }

    fn bpe_encode_chunk(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        // Map raw UTF-8 bytes to GPT-2 byte-level unicode characters, then look up single-char IDs.
        let mut ids: Vec<u32> = Vec::with_capacity(text.len());
        let mut tmp = [0u8; 4];
        for b in text.bytes() {
            let ch = self.byte_to_char[b as usize];
            let s = ch.encode_utf8(&mut tmp);
            if let Some(&id) = self.token_to_id.get(s) {
                ids.push(id);
            }
        }

        // Iteratively apply the highest-priority (lowest-rank) BPE merge.
        loop {
            if ids.len() < 2 {
                break;
            }
            let mut best_rank = u32::MAX;
            let mut best_idx = usize::MAX;
            for i in 0..(ids.len() - 1) {
                if let Some(&rank) = self.merges.get(&(ids[i], ids[i + 1])) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }
            if best_idx == usize::MAX {
                break;
            }
            let mut merged_str = String::with_capacity(
                self.vocab[ids[best_idx] as usize].len()
                    + self.vocab[ids[best_idx + 1] as usize].len(),
            );
            merged_str.push_str(&self.vocab[ids[best_idx] as usize]);
            merged_str.push_str(&self.vocab[ids[best_idx + 1] as usize]);
            if let Some(&new_id) = self.token_to_id.get(&merged_str) {
                ids[best_idx] = new_id;
                ids.remove(best_idx + 1);
            } else {
                break;
            }
        }
        out.extend_from_slice(&ids);
    }

    /// Splits a plain text segment into GPT-2/Qwen-style pre-tokens (words with
    /// optional leading space, digits, punctuation, newlines) before BPE merges.
    fn pretokenize_and_encode(&self, text: &str, out: &mut Vec<u32>) {
        let bytes = text.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            let start = i;
            let b = bytes[i];
            if b == b'\n' || b == b'\r' {
                while i < bytes.len() && (bytes[i] == b'\n' || bytes[i] == b'\r') {
                    i += 1;
                }
            } else if b == b' ' {
                i += 1;
                if i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                    while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                        i += 1;
                    }
                } else if i < bytes.len() && bytes[i] == b' ' {
                    while i < bytes.len() && bytes[i] == b' ' {
                        i += 1;
                    }
                } else if i < bytes.len() && !bytes[i].is_ascii_alphanumeric() && bytes[i] != b'\n' && bytes[i] != b'\r' {
                    i += 1;
                }
            } else if b.is_ascii_alphabetic() || b >= 0x80 {
                while i < bytes.len() && (bytes[i].is_ascii_alphabetic() || bytes[i] >= 0x80) {
                    i += 1;
                }
            } else if b.is_ascii_digit() {
                // Qwen3 splits digits individually
                i += 1;
            } else {
                i += 1;
            }
            if let Ok(chunk) = core::str::from_utf8(&bytes[start..i]) {
                self.bpe_encode_chunk(chunk, out);
            }
        }
    }

    /// Encodes `text`, matching special tokens verbatim and BPE-encoding everything else.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // Find earliest occurring special token.
            let mut earliest: Option<(usize, usize, u32)> = None;
            for (tok, id) in &self.special_tokens {
                if let Some(pos) = rest.find(tok.as_str()) {
                    let replace = match earliest {
                        None => true,
                        Some((best_pos, best_len, _)) => {
                            pos < best_pos || (pos == best_pos && tok.len() > best_len)
                        }
                    };
                    if replace {
                        earliest = Some((pos, tok.len(), *id));
                    }
                }
            }
            match earliest {
                Some((pos, len, id)) => {
                    if pos > 0 {
                        self.pretokenize_and_encode(&rest[..pos], &mut out);
                    }
                    out.push(id);
                    rest = &rest[pos + len..];
                }
                None => {
                    self.pretokenize_and_encode(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    /// Formats a multi-turn conversation in Qwen3 ChatML format.
    /// When `enable_thinking` is false, appends `<think>\n\n</think>\n\n` so Qwen3
    /// responds directly without burning tokens on internal monologue.
    pub fn encode_chat_turns(
        &self,
        system_prompt: &str,
        turns: &[(bool, &str)], // (is_user, text)
        enable_thinking: bool,
    ) -> Vec<u32> {
        let mut prompt = String::new();
        if self.im_start_id.is_some() {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(system_prompt);
            prompt.push_str("<|im_end|>\n");
            for &(is_user, text) in turns {
                if is_user {
                    prompt.push_str("<|im_start|>user\n");
                } else {
                    prompt.push_str("<|im_start|>assistant\n");
                }
                prompt.push_str(text);
                prompt.push_str("<|im_end|>\n");
            }
            prompt.push_str("<|im_start|>assistant\n");
            if !enable_thinking && self.think_start_id.is_some() && self.think_end_id.is_some() {
                prompt.push_str("<think>\n\n</think>\n\n");
            }
        } else {
            // Generic Llama-3 / plain fallback
            for &(is_user, text) in turns {
                if is_user {
                    prompt.push_str("User: ");
                } else {
                    prompt.push_str("Assistant: ");
                }
                prompt.push_str(text);
                prompt.push('\n');
            }
            prompt.push_str("Assistant:");
        }
        self.encode(&prompt)
    }

    /// Decodes a single token ID to raw UTF-8 bytes.
    pub fn decode_token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(tok) = self.vocab.get(id as usize) else {
            return Vec::new();
        };
        if self.special_tokens.iter().any(|(_, sid)| *sid == id) {
            return Vec::new();
        }
        let mut bytes = Vec::with_capacity(tok.len());
        for ch in tok.chars() {
            if let Some(&b) = self.char_to_byte.get(&ch) {
                bytes.push(b);
            } else {
                let mut buf = [0u8; 4];
                bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
        bytes
    }

    pub fn token_str_raw(&self, id: u32) -> &str {
        self.vocab.get(id as usize).map(|s| s.as_str()).unwrap_or("")
    }
}

/// Buffers incomplete trailing UTF-8 byte sequences across streamed tokens.
#[derive(Default)]
pub struct Utf8StreamDecoder {
    pending: Vec<u8>,
}

impl Utf8StreamDecoder {
    pub fn push_bytes(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match core::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    out.push_str(valid);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid_len = e.valid_up_to();
                    if valid_len > 0 {
                        if let Ok(s) = core::str::from_utf8(&self.pending[..valid_len]) {
                            out.push_str(s);
                        }
                        self.pending.drain(..valid_len);
                    }
                    if let Some(err_len) = e.error_len() {
                        // Invalid byte sequence: skip and emit replacement char.
                        out.push('\u{FFFD}');
                        self.pending.drain(..err_len);
                    } else {
                        // Incomplete multi-byte char at end: wait for next token.
                        break;
                    }
                }
            }
        }
        out
    }

    pub fn clear(&mut self) {
        self.pending.clear();
    }
}
