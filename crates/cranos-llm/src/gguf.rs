use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Bf16 = 30,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            30 => Some(Self::Bf16),
            _ => None,
        }
    }

    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }

    pub fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
        }
    }

    pub fn row_bytes(self, cols: usize) -> usize {
        (cols / self.block_size()) * self.type_size()
    }
}

#[derive(Clone, Debug)]
pub enum GgufValue<'a> {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(&'a str),
    U64(u64),
    I64(i64),
    F64(f64),
    StrArray(Vec<&'a str>),
    I32Array(Vec<i32>),
    U32Array(Vec<u32>),
    F32Array(Vec<f32>),
    SkippedArray,
}

impl<'a> GgufValue<'a> {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(v as u64),
            Self::U16(v) => Some(v as u64),
            Self::U32(v) => Some(v as u64),
            Self::U64(v) => Some(v),
            Self::I32(v) if v >= 0 => Some(v as u64),
            Self::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            Self::F64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&'a str> {
        match *self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dims: [usize; 4],
    pub n_dims: u32,
    pub ty: GgmlType,
    pub data_offset: usize,
    pub byte_len: usize,
}

pub struct GgufFile<'a> {
    pub data: &'a [u8],
    pub version: u32,
    pub kv: BTreeMap<String, GgufValue<'a>>,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub data_base: usize,
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], &'static str> {
        let end = self.pos.checked_add(n).ok_or("gguf offset overflow")?;
        if end > self.data.len() {
            return Err("unexpected EOF in GGUF");
        }
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, &'static str> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, &'static str> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, &'static str> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, &'static str> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn str(&mut self) -> Result<&'a str, &'static str> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        // Token strings in GGUF are valid UTF-8 (GPT-2 byte-level BPE).
        Ok(core::str::from_utf8(bytes).unwrap_or(""))
    }

    fn read_value(&mut self, ty: u32, keep_arrays: bool) -> Result<GgufValue<'a>, &'static str> {
        match ty {
            0 => Ok(GgufValue::U8(self.u8()?)),
            1 => Ok(GgufValue::I8(self.u8()? as i8)),
            2 => Ok(GgufValue::U16(self.u16()?)),
            3 => Ok(GgufValue::I16(self.u16()? as i16)),
            4 => Ok(GgufValue::U32(self.u32()?)),
            5 => Ok(GgufValue::I32(self.u32()? as i32)),
            6 => Ok(GgufValue::F32(f32::from_bits(self.u32()?))),
            7 => Ok(GgufValue::Bool(self.u8()? != 0)),
            8 => Ok(GgufValue::Str(self.str()?)),
            10 => Ok(GgufValue::U64(self.u64()?)),
            11 => Ok(GgufValue::I64(self.u64()? as i64)),
            12 => Ok(GgufValue::F64(f64::from_bits(self.u64()?))),
            9 => {
                let elem_ty = self.u32()?;
                let count = self.u64()? as usize;
                match elem_ty {
                    8 if keep_arrays => {
                        let mut v = Vec::with_capacity(count);
                        for _ in 0..count {
                            v.push(self.str()?);
                        }
                        Ok(GgufValue::StrArray(v))
                    }
                    5 if keep_arrays => {
                        let mut v = Vec::with_capacity(count);
                        for _ in 0..count {
                            v.push(self.u32()? as i32);
                        }
                        Ok(GgufValue::I32Array(v))
                    }
                    4 if keep_arrays => {
                        let mut v = Vec::with_capacity(count);
                        for _ in 0..count {
                            v.push(self.u32()?);
                        }
                        Ok(GgufValue::U32Array(v))
                    }
                    6 if keep_arrays => {
                        let mut v = Vec::with_capacity(count);
                        for _ in 0..count {
                            v.push(f32::from_bits(self.u32()?));
                        }
                        Ok(GgufValue::F32Array(v))
                    }
                    _ => {
                        self.skip_array_elements(elem_ty, count)?;
                        Ok(GgufValue::SkippedArray)
                    }
                }
            }
            _ => Err("unknown GGUF value type"),
        }
    }

    fn skip_array_elements(&mut self, elem_ty: u32, count: usize) -> Result<(), &'static str> {
        match elem_ty {
            0 | 1 | 7 => {
                let _ = self.take(count)?;
            }
            2 | 3 => {
                let _ = self.take(count * 2)?;
            }
            4 | 5 | 6 => {
                let _ = self.take(count * 4)?;
            }
            10 | 11 | 12 => {
                let _ = self.take(count * 8)?;
            }
            8 => {
                for _ in 0..count {
                    let len = self.u64()? as usize;
                    let _ = self.take(len)?;
                }
            }
            _ => return Err("unsupported GGUF nested array type"),
        }
        Ok(())
    }
}

impl<'a> GgufFile<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, &'static str> {
        let mut c = Cursor::new(data);
        if c.take(4)? != GGUF_MAGIC {
            return Err("invalid GGUF magic");
        }
        let version = c.u32()?;
        if version < 2 || version > 3 {
            return Err("unsupported GGUF version");
        }
        let tensor_count = c.u64()? as usize;
        let kv_count = c.u64()? as usize;

        let mut kv = BTreeMap::new();
        for _ in 0..kv_count {
            let key = c.str()?;
            let ty = c.u32()?;
            let keep = key == "tokenizer.ggml.tokens"
                || key == "tokenizer.ggml.merges"
                || key == "tokenizer.ggml.token_type"
                || key == "tokenizer.ggml.scores";
            let val = c.read_value(ty, keep)?;
            kv.insert(String::from(key), val);
        }

        let alignment = kv
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;

        let mut raw_tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = String::from(c.str()?);
            let n_dims = c.u32()?;
            if n_dims == 0 || n_dims > 4 {
                return Err("invalid tensor n_dims");
            }
            let mut dims = [1usize; 4];
            for d in 0..n_dims as usize {
                dims[d] = c.u64()? as usize;
            }
            let ty_raw = c.u32()?;
            let ty = GgmlType::from_u32(ty_raw).ok_or("unsupported GGML tensor type")?;
            let rel_offset = c.u64()? as usize;

            let n_rows = dims[1].saturating_mul(dims[2]).saturating_mul(dims[3]);
            let byte_len = ty.row_bytes(dims[0]).saturating_mul(n_rows);
            raw_tensors.push((
                name,
                TensorInfo {
                    dims,
                    n_dims,
                    ty,
                    data_offset: rel_offset,
                    byte_len,
                },
            ));
        }

        let data_base = (c.pos + alignment - 1) & !(alignment - 1);
        let mut tensors = BTreeMap::new();
        for (name, mut info) in raw_tensors {
            let abs_off = data_base
                .checked_add(info.data_offset)
                .ok_or("tensor offset overflow")?;
            if abs_off.saturating_add(info.byte_len) > data.len() {
                return Err("tensor extends past end of GGUF");
            }
            info.data_offset = abs_off;
            tensors.insert(name, info);
        }

        Ok(Self {
            data,
            version,
            kv,
            tensors,
            data_base,
        })
    }

    pub fn tensor_bytes(&self, name: &str) -> Option<(&TensorInfo, &'a [u8])> {
        let info = self.tensors.get(name)?;
        let slice = &self.data[info.data_offset..info.data_offset + info.byte_len];
        Some((info, slice))
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.kv.get(key).and_then(|v| v.as_u64())
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.kv.get(key).and_then(|v| v.as_f32())
    }

    pub fn get_str(&self, key: &str) -> Option<&'a str> {
        self.kv.get(key).and_then(|v| v.as_str())
    }
}
