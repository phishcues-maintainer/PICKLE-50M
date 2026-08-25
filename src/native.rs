use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;

use crate::sha256;
use crate::tokenizer::Tokenizer;

const MAGIC_V3: &[u8; 8] = b"PKNATV3\0";
const MAGIC_V4: &[u8; 8] = b"PKNATV4\0";
const FLAG_TIED_EMBEDDINGS: u32 = 1 << 0;
const FLAG_DEFAULT_ADD_BOS: u32 = 1 << 1;
const KNOWN_V4_FLAGS: u32 = FLAG_TIED_EMBEDDINGS | FLAG_DEFAULT_ADD_BOS;
const Q2: [f32; 4] = [-1.510418, -0.452780, 0.452780, 1.510418];
const Q3: [f32; 8] = [
    -2.151933, -1.343899, -0.755999, -0.2450922, 0.2450922, 0.755999, 1.343899, 2.151933,
];
const Q4: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.39491749,
    -0.28444138,
    -0.18477343,
    -0.09105,
    0.0,
    0.0795803,
    0.1609302,
    0.2461123,
    0.3379152,
    0.4407098,
    0.562617,
    0.7229568,
    1.0,
];

#[derive(Clone, Copy)]
enum Kind {
    Q2,
    Q2Symmetric,
    Q3,
    Q4,
}

#[derive(Clone, Copy)]
struct Tensor {
    rows: usize,
    cols: usize,
    groups: usize,
    offset: usize,
    codes_offset: usize,
    kind: Kind,
}

struct Layer {
    input_norm: Vec<f32>,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    post_norm: Vec<f32>,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

pub struct Model {
    data: Vec<u8>,
    pub group_size: usize,
    pub layers_count: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab_size: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub context: usize,
    pub bos_token: u32,
    #[allow(dead_code)]
    pub eos_token: u32,
    #[allow(dead_code)]
    pub pad_token: u32,
    pub format_version: u32,
    pub tied_embeddings: bool,
    pub default_add_bos: bool,
    rms_eps: f32,
    rope_theta: f32,
    embed: Tensor,
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
    lm_head: Tensor,
    tokenizer: Tokenizer,
    use_avx2: bool,
    authenticated_digest: [u8; 32],
}

pub struct State {
    max_seq: usize,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
    x: Vec<f32>,
    xb: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    logits: Vec<f32>,
    scores: Vec<f32>,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn read_u32(data: &[u8], cursor: &mut usize) -> io::Result<u32> {
    let bytes = data
        .get(*cursor..*cursor + 4)
        .ok_or_else(|| invalid("truncated native model header"))?;
    *cursor += 4;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_f32(data: &[u8], cursor: &mut usize) -> io::Result<f32> {
    Ok(f32::from_bits(read_u32(data, cursor)?))
}

fn read_u64(data: &[u8], cursor: &mut usize) -> io::Result<u64> {
    let bytes = data
        .get(*cursor..*cursor + 8)
        .ok_or_else(|| invalid("truncated native model header"))?;
    *cursor += 8;
    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
}

fn take_kind(data: &[u8], cursor: &mut usize) -> io::Result<u8> {
    let kind = *data
        .get(*cursor)
        .ok_or_else(|| invalid("truncated tensor kind"))?;
    *cursor += 1;
    Ok(kind)
}

#[inline]
fn half_to_f32(value: u16) -> f32 {
    let sign = ((value & 0x8000) as u32) << 16;
    let exponent = ((value >> 10) & 0x1f) as u32;
    let mantissa = (value & 0x03ff) as u32;
    let bits = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut mant = mantissa;
            let mut exp = 113_u32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                exp -= 1;
            }
            sign | (exp << 23) | ((mant & 0x03ff) << 13)
        }
    } else if exponent == 31 {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 112) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}

fn take_norm(data: &[u8], cursor: &mut usize, width: usize) -> io::Result<Vec<f32>> {
    if take_kind(data, cursor)? != 0 {
        return Err(invalid("normalization tensor must be fp16"));
    }
    let bytes = width
        .checked_mul(2)
        .ok_or_else(|| invalid("tensor size overflow"))?;
    let end = cursor
        .checked_add(bytes)
        .ok_or_else(|| invalid("tensor size overflow"))?;
    let raw = data
        .get(*cursor..end)
        .ok_or_else(|| invalid("truncated fp16 norm tensor"))?;
    let mut out = Vec::with_capacity(width);
    for bytes in raw.chunks_exact(2) {
        out.push(half_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])));
    }
    *cursor = end;
    Ok(out)
}

fn take_tensor(
    data: &[u8],
    cursor: &mut usize,
    rows: usize,
    cols: usize,
    group_size: usize,
) -> io::Result<Tensor> {
    let kind = match take_kind(data, cursor)? {
        2 => Kind::Q2,
        3 => Kind::Q2Symmetric,
        4 => Kind::Q4,
        5 => Kind::Q3,
        _ => return Err(invalid("quantized tensor has unsupported kind")),
    };
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid("tensor element count overflow"))?;
    if count % group_size != 0 || cols % group_size != 0 {
        return Err(invalid(
            "native runtime requires row-aligned quantization groups",
        ));
    }
    let groups = count / group_size;
    let packed = match kind {
        Kind::Q2 | Kind::Q2Symmetric => count.div_ceil(4),
        Kind::Q3 => count.div_ceil(8) * 3,
        Kind::Q4 => count.div_ceil(2),
    };
    let scale_bytes = groups
        * match kind {
            Kind::Q2Symmetric => 4,
            Kind::Q2 | Kind::Q3 | Kind::Q4 => 2,
        };
    let end = cursor
        .checked_add(scale_bytes + packed)
        .ok_or_else(|| invalid("tensor byte size overflow"))?;
    if end > data.len() {
        return Err(invalid("truncated quantized tensor"));
    }
    let tensor = Tensor {
        rows,
        cols,
        groups,
        offset: *cursor,
        codes_offset: *cursor + scale_bytes,
        kind,
    };
    *cursor = end;
    Ok(tensor)
}

impl Model {
    pub fn load(path: &Path) -> io::Result<Self> {
        let data = fs::read(path)?;
        let format_version = match data.get(..8) {
            Some(magic) if magic == MAGIC_V3 => 3,
            Some(magic) if magic == MAGIC_V4 => 4,
            _ => return Err(invalid("not a supported PICKLE native model")),
        };
        let mut cursor = 8;
        let group_size = read_u32(&data, &mut cursor)? as usize;
        let layers_count = read_u32(&data, &mut cursor)? as usize;
        let hidden = read_u32(&data, &mut cursor)? as usize;
        let intermediate = read_u32(&data, &mut cursor)? as usize;
        let vocab_size = read_u32(&data, &mut cursor)? as usize;
        let heads = read_u32(&data, &mut cursor)? as usize;
        let kv_heads = read_u32(&data, &mut cursor)? as usize;
        let head_dim = read_u32(&data, &mut cursor)? as usize;
        let context = read_u32(&data, &mut cursor)? as usize;
        let bos_token = read_u32(&data, &mut cursor)?;
        let eos_token = read_u32(&data, &mut cursor)?;
        let pad_token = read_u32(&data, &mut cursor)?;
        let rms_eps = read_f32(&data, &mut cursor)?;
        let rope_theta = read_f32(&data, &mut cursor)?;
        let flags = if format_version >= 4 {
            read_u32(&data, &mut cursor)?
        } else {
            FLAG_DEFAULT_ADD_BOS
        };
        if flags & !KNOWN_V4_FLAGS != 0 {
            return Err(invalid("native model uses unsupported feature flags"));
        }
        let tied_embeddings = flags & FLAG_TIED_EMBEDDINGS != 0;
        let default_add_bos = flags & FLAG_DEFAULT_ADD_BOS != 0;
        let body_bytes = usize::try_from(read_u64(&data, &mut cursor)?)
            .map_err(|_| invalid("native model body is too large"))?;
        let digest_offset = cursor;
        let expected_digest: [u8; 32] = data
            .get(cursor..cursor + 32)
            .ok_or_else(|| invalid("truncated native model checksum"))?
            .try_into()
            .unwrap();
        cursor += 32;
        if data.len().checked_sub(cursor) != Some(body_bytes) {
            return Err(invalid("native model body length mismatch"));
        }
        let body_digest = sha256::digest(&data[cursor..]);
        let mut authentication_material = Vec::with_capacity(digest_offset + body_digest.len());
        authentication_material.extend_from_slice(&data[..digest_offset]);
        authentication_material.extend_from_slice(&body_digest);
        if sha256::digest(&authentication_material) != expected_digest {
            return Err(invalid("native model SHA-256 checksum mismatch"));
        }

        if group_size == 0
            || group_size > 4096
            || layers_count == 0
            || layers_count > 1024
            || hidden == 0
            || hidden > 65536
            || intermediate == 0
            || intermediate > 262144
            || vocab_size == 0
            || vocab_size > 1_000_000
            || heads == 0
            || kv_heads == 0
            || head_dim == 0
            || context == 0
            || context > 1_000_000
            || heads.checked_mul(head_dim) != Some(hidden)
            || heads % kv_heads != 0
            || hidden % group_size != 0
            || intermediate % group_size != 0
        {
            return Err(invalid(
                "unsupported or inconsistent native model dimensions",
            ));
        }
        let kv_width = kv_heads * head_dim;
        let embed = take_tensor(&data, &mut cursor, vocab_size, hidden, group_size)?;
        let mut layers = Vec::with_capacity(layers_count);
        for _ in 0..layers_count {
            let input_norm = take_norm(&data, &mut cursor, hidden)?;
            let q = take_tensor(&data, &mut cursor, hidden, hidden, group_size)?;
            let k = take_tensor(&data, &mut cursor, kv_width, hidden, group_size)?;
            let v = take_tensor(&data, &mut cursor, kv_width, hidden, group_size)?;
            let o = take_tensor(&data, &mut cursor, hidden, hidden, group_size)?;
            let post_norm = take_norm(&data, &mut cursor, hidden)?;
            let gate = take_tensor(&data, &mut cursor, intermediate, hidden, group_size)?;
            let up = take_tensor(&data, &mut cursor, intermediate, hidden, group_size)?;
            let down = take_tensor(&data, &mut cursor, hidden, intermediate, group_size)?;
            layers.push(Layer {
                input_norm,
                q,
                k,
                v,
                o,
                post_norm,
                gate,
                up,
                down,
            });
        }
        let final_norm = take_norm(&data, &mut cursor, hidden)?;
        let lm_head = if tied_embeddings {
            embed
        } else {
            take_tensor(&data, &mut cursor, vocab_size, hidden, group_size)?
        };

        let tokenizer_bytes = read_u32(&data, &mut cursor)? as usize;
        let end = cursor
            .checked_add(tokenizer_bytes)
            .ok_or_else(|| invalid("native tokenizer length overflow"))?;
        let tokenizer_data = data
            .get(cursor..end)
            .ok_or_else(|| invalid("truncated native tokenizer"))?;
        let tokenizer = Tokenizer::load(tokenizer_data, vocab_size)?;
        cursor = end;
        if cursor != data.len() {
            return Err(invalid("trailing bytes in native model"));
        }

        Ok(Self {
            data,
            group_size,
            layers_count,
            hidden,
            intermediate,
            vocab_size,
            heads,
            kv_heads,
            head_dim,
            context,
            bos_token,
            eos_token,
            pad_token,
            format_version,
            tied_embeddings,
            default_add_bos,
            rms_eps,
            rope_theta,
            embed,
            layers,
            final_norm,
            lm_head,
            tokenizer,
            use_avx2: avx2_available(),
            authenticated_digest: expected_digest,
        })
    }

    pub fn bytes(&self) -> usize {
        self.data.len()
    }

    pub fn state(&self, max_seq: usize) -> io::Result<State> {
        if max_seq == 0 || max_seq > self.context {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("sequence capacity must be between 1 and {}", self.context),
            ));
        }
        let kv_width = self.kv_heads * self.head_dim;
        let cache_values = self
            .layers_count
            .checked_mul(max_seq)
            .and_then(|value| value.checked_mul(kv_width))
            .ok_or_else(|| invalid("KV cache size overflow"))?;
        Ok(State {
            max_seq,
            key_cache: vec![0.0; cache_values],
            value_cache: vec![0.0; cache_values],
            x: vec![0.0; self.hidden],
            xb: vec![0.0; self.hidden],
            q: vec![0.0; self.hidden],
            k: vec![0.0; kv_width],
            v: vec![0.0; kv_width],
            att: vec![0.0; self.hidden],
            gate: vec![0.0; self.intermediate],
            up: vec![0.0; self.intermediate],
            logits: vec![0.0; self.vocab_size],
            scores: vec![0.0; max_seq],
        })
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens)
    }

    pub fn tokenizer_name(&self) -> &'static str {
        self.tokenizer.name()
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let encoded = self.tokenizer.encode(text);
        if add_bos {
            let mut with_bos = Vec::with_capacity(encoded.len() + 1);
            with_bos.push(self.bos_token);
            with_bos.extend(encoded);
            with_bos
        } else {
            encoded
        }
    }

    pub fn set_kernel(&mut self, value: &str) -> io::Result<()> {
        self.use_avx2 = match value {
            "auto" => avx2_available(),
            "scalar" => false,
            "avx2" if avx2_available() => true,
            "avx2" => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "AVX2 was requested but is unavailable on this CPU",
                ))
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "--kernel must be auto, scalar, or avx2",
                ))
            }
        };
        Ok(())
    }

    pub fn kernel_name(&self) -> &'static str {
        if self.use_avx2 {
            "avx2"
        } else {
            "scalar"
        }
    }

    pub fn authenticated_sha256(&self) -> String {
        sha256::hex(&self.authenticated_digest)
    }

    pub fn forward<'a>(
        &self,
        state: &'a mut State,
        token: u32,
        position: usize,
    ) -> io::Result<&'a [f32]> {
        if token as usize >= self.vocab_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("token ID {token} is outside vocabulary"),
            ));
        }
        if position >= state.max_seq {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "position exceeds allocated sequence capacity",
            ));
        }
        dequant_row(
            &self.data,
            self.embed,
            token as usize,
            self.group_size,
            &mut state.x,
        );
        let kv_width = self.kv_heads * self.head_dim;

        for (layer_index, layer) in self.layers.iter().enumerate() {
            rms_norm(&state.x, &layer.input_norm, self.rms_eps, &mut state.xb);
            matvec(
                &self.data,
                layer.q,
                self.group_size,
                &state.xb,
                &mut state.q,
                self.use_avx2,
            );
            matvec(
                &self.data,
                layer.k,
                self.group_size,
                &state.xb,
                &mut state.k,
                self.use_avx2,
            );
            matvec(
                &self.data,
                layer.v,
                self.group_size,
                &state.xb,
                &mut state.v,
                self.use_avx2,
            );
            rope(
                &mut state.q,
                self.heads,
                self.head_dim,
                position,
                self.rope_theta,
            );
            rope(
                &mut state.k,
                self.kv_heads,
                self.head_dim,
                position,
                self.rope_theta,
            );

            let cache_start = (layer_index * state.max_seq + position) * kv_width;
            state.key_cache[cache_start..cache_start + kv_width].copy_from_slice(&state.k);
            state.value_cache[cache_start..cache_start + kv_width].copy_from_slice(&state.v);
            attention(
                &state.q,
                &state.key_cache,
                &state.value_cache,
                layer_index,
                position,
                state.max_seq,
                self.heads,
                self.kv_heads,
                self.head_dim,
                &mut state.att,
                &mut state.scores,
            );
            matvec(
                &self.data,
                layer.o,
                self.group_size,
                &state.att,
                &mut state.xb,
                self.use_avx2,
            );
            for (x, residual) in state.x.iter_mut().zip(&state.xb) {
                *x += residual;
            }

            rms_norm(&state.x, &layer.post_norm, self.rms_eps, &mut state.xb);
            matvec(
                &self.data,
                layer.gate,
                self.group_size,
                &state.xb,
                &mut state.gate,
                self.use_avx2,
            );
            matvec(
                &self.data,
                layer.up,
                self.group_size,
                &state.xb,
                &mut state.up,
                self.use_avx2,
            );
            for (gate, up) in state.gate.iter_mut().zip(&state.up) {
                *gate = *gate / (1.0 + (-*gate).exp()) * up;
            }
            matvec(
                &self.data,
                layer.down,
                self.group_size,
                &state.gate,
                &mut state.xb,
                self.use_avx2,
            );
            for (x, residual) in state.x.iter_mut().zip(&state.xb) {
                *x += residual;
            }
        }

        rms_norm(&state.x, &self.final_norm, self.rms_eps, &mut state.xb);
        matvec(
            &self.data,
            self.lm_head,
            self.group_size,
            &state.xb,
            &mut state.logits,
            self.use_avx2,
        );
        Ok(&state.logits)
    }
}

impl State {
    pub fn allocated_bytes(&self) -> usize {
        (self.key_cache.len()
            + self.value_cache.len()
            + self.x.len()
            + self.xb.len()
            + self.q.len()
            + self.k.len()
            + self.v.len()
            + self.att.len()
            + self.gate.len()
            + self.up.len()
            + self.logits.len()
            + self.scores.len())
            * std::mem::size_of::<f32>()
    }
}

#[inline]
fn scale(data: &[u8], tensor: Tensor, group: usize) -> f32 {
    let offset = tensor.offset + group * 2;
    half_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]))
}

#[inline]
fn q2_levels(data: &[u8], tensor: Tensor, group: usize) -> [f32; 4] {
    if !matches!(tensor.kind, Kind::Q2Symmetric) {
        return Q2;
    }
    let offset = tensor.offset + group * 4;
    let inner = half_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]));
    let outer = half_to_f32(u16::from_le_bytes([data[offset + 2], data[offset + 3]]));
    [-outer, -inner, inner, outer]
}

fn dequant_row(data: &[u8], tensor: Tensor, row: usize, group_size: usize, output: &mut [f32]) {
    debug_assert!(row < tensor.rows && output.len() == tensor.cols);
    let groups_per_row = tensor.cols / group_size;
    let packed_per_group = match tensor.kind {
        Kind::Q2 | Kind::Q2Symmetric => group_size / 4,
        Kind::Q3 => group_size / 8 * 3,
        Kind::Q4 => group_size / 2,
    };
    for group_in_row in 0..groups_per_row {
        let group = row * groups_per_row + group_in_row;
        let factor = if matches!(tensor.kind, Kind::Q2Symmetric) {
            1.0
        } else {
            scale(data, tensor, group)
        };
        let code_start = tensor.codes_offset + group * packed_per_group;
        let out_start = group_in_row * group_size;
        match tensor.kind {
            Kind::Q2 | Kind::Q2Symmetric => {
                let levels = q2_levels(data, tensor, group);
                for packed_index in 0..packed_per_group {
                    let byte = data[code_start + packed_index];
                    let base = out_start + packed_index * 4;
                    output[base] = factor * levels[(byte & 3) as usize];
                    output[base + 1] = factor * levels[((byte >> 2) & 3) as usize];
                    output[base + 2] = factor * levels[((byte >> 4) & 3) as usize];
                    output[base + 3] = factor * levels[(byte >> 6) as usize];
                }
            }
            Kind::Q4 => {
                for packed_index in 0..packed_per_group {
                    let byte = data[code_start + packed_index];
                    let base = out_start + packed_index * 2;
                    output[base] = factor * Q4[(byte & 15) as usize];
                    output[base + 1] = factor * Q4[(byte >> 4) as usize];
                }
            }
            Kind::Q3 => {
                for index in 0..group_size {
                    let bit = index * 3;
                    let byte_index = bit / 8;
                    let shift = bit % 8;
                    let low = data[code_start + byte_index] as u16;
                    let high = if byte_index + 1 < packed_per_group {
                        (data[code_start + byte_index + 1] as u16) << 8
                    } else {
                        0
                    };
                    output[out_start + index] = factor * Q3[(((low | high) >> shift) & 7) as usize];
                }
            }
        }
    }
}

fn scalar_group_sum(kind: Kind, codes: &[u8], input: &[f32], q2: &[f32; 4]) -> f32 {
    let mut sum = 0.0_f32;
    match kind {
        Kind::Q2 | Kind::Q2Symmetric => {
            for (packed_index, &byte) in codes.iter().enumerate() {
                let base = packed_index * 4;
                sum += input[base] * q2[(byte & 3) as usize]
                    + input[base + 1] * q2[((byte >> 2) & 3) as usize]
                    + input[base + 2] * q2[((byte >> 4) & 3) as usize]
                    + input[base + 3] * q2[(byte >> 6) as usize];
            }
        }
        Kind::Q4 => {
            for (packed_index, &byte) in codes.iter().enumerate() {
                let base = packed_index * 2;
                sum += input[base] * Q4[(byte & 15) as usize]
                    + input[base + 1] * Q4[(byte >> 4) as usize];
            }
        }
        Kind::Q3 => {
            for (chunk, values) in codes.chunks_exact(3).zip(input.chunks_exact(8)) {
                let word = chunk[0] as u32 | (chunk[1] as u32) << 8 | (chunk[2] as u32) << 16;
                for (index, &value) in values.iter().enumerate() {
                    sum += value * Q3[((word >> (index * 3)) & 7) as usize];
                }
            }
        }
    }
    sum
}

#[inline]
fn avx2_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[cfg(target_arch = "x86")]
use std::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as arch;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx2_group_sum(kind: Kind, codes: &[u8], input: &[f32], q2: &[f32; 4]) -> f32 {
    let mut accumulator = arch::_mm256_setzero_ps();
    match kind {
        Kind::Q2 | Kind::Q2Symmetric => {
            debug_assert_eq!(input.len(), codes.len() * 4);
            for packed in 0..codes.len() / 2 {
                let first = codes[packed * 2];
                let second = codes[packed * 2 + 1];
                let indices = arch::_mm256_setr_epi32(
                    (first & 3) as i32,
                    ((first >> 2) & 3) as i32,
                    ((first >> 4) & 3) as i32,
                    (first >> 6) as i32,
                    (second & 3) as i32,
                    ((second >> 2) & 3) as i32,
                    ((second >> 4) & 3) as i32,
                    (second >> 6) as i32,
                );
                let levels = arch::_mm256_i32gather_ps(q2.as_ptr(), indices, 4);
                let values = arch::_mm256_loadu_ps(input.as_ptr().add(packed * 8));
                accumulator = arch::_mm256_add_ps(accumulator, arch::_mm256_mul_ps(values, levels));
            }
        }
        Kind::Q4 => {
            debug_assert_eq!(input.len(), codes.len() * 2);
            for packed in 0..codes.len() / 4 {
                let a = codes[packed * 4];
                let b = codes[packed * 4 + 1];
                let c = codes[packed * 4 + 2];
                let d = codes[packed * 4 + 3];
                let indices = arch::_mm256_setr_epi32(
                    (a & 15) as i32,
                    (a >> 4) as i32,
                    (b & 15) as i32,
                    (b >> 4) as i32,
                    (c & 15) as i32,
                    (c >> 4) as i32,
                    (d & 15) as i32,
                    (d >> 4) as i32,
                );
                let levels = arch::_mm256_i32gather_ps(Q4.as_ptr(), indices, 4);
                let values = arch::_mm256_loadu_ps(input.as_ptr().add(packed * 8));
                accumulator = arch::_mm256_add_ps(accumulator, arch::_mm256_mul_ps(values, levels));
            }
        }
        Kind::Q3 => {
            debug_assert_eq!(input.len() / 8 * 3, codes.len());
            for (packed, chunk) in codes.chunks_exact(3).enumerate() {
                let word = chunk[0] as u32 | (chunk[1] as u32) << 8 | (chunk[2] as u32) << 16;
                let indices = arch::_mm256_setr_epi32(
                    (word & 7) as i32,
                    ((word >> 3) & 7) as i32,
                    ((word >> 6) & 7) as i32,
                    ((word >> 9) & 7) as i32,
                    ((word >> 12) & 7) as i32,
                    ((word >> 15) & 7) as i32,
                    ((word >> 18) & 7) as i32,
                    ((word >> 21) & 7) as i32,
                );
                let levels = arch::_mm256_i32gather_ps(Q3.as_ptr(), indices, 4);
                let values = arch::_mm256_loadu_ps(input.as_ptr().add(packed * 8));
                accumulator = arch::_mm256_add_ps(accumulator, arch::_mm256_mul_ps(values, levels));
            }
        }
    }
    let low = arch::_mm256_castps256_ps128(accumulator);
    let high = arch::_mm256_extractf128_ps(accumulator, 1);
    let sum4 = arch::_mm_add_ps(low, high);
    let pair = arch::_mm_add_ps(sum4, arch::_mm_movehl_ps(sum4, sum4));
    let total = arch::_mm_add_ss(pair, arch::_mm_shuffle_ps(pair, pair, 0x55));
    arch::_mm_cvtss_f32(total)
}

fn row_dot(
    data: &[u8],
    tensor: Tensor,
    group_size: usize,
    input: &[f32],
    row: usize,
    use_avx2: bool,
) -> f32 {
    let groups_per_row = tensor.cols / group_size;
    let packed_per_group = match tensor.kind {
        Kind::Q2 | Kind::Q2Symmetric => group_size / 4,
        Kind::Q3 => group_size / 8 * 3,
        Kind::Q4 => group_size / 2,
    };
    let mut sum = 0.0_f32;
    for group_in_row in 0..groups_per_row {
        let group = row * groups_per_row + group_in_row;
        let factor = if matches!(tensor.kind, Kind::Q2Symmetric) {
            1.0
        } else {
            scale(data, tensor, group)
        };
        let q2 = q2_levels(data, tensor, group);
        let code_start = tensor.codes_offset + group * packed_per_group;
        let codes = &data[code_start..code_start + packed_per_group];
        let input_start = group_in_row * group_size;
        let values = &input[input_start..input_start + group_size];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let group_sum = if use_avx2 {
            // SAFETY: runtime feature detection guarantees AVX2 and slices have exact group sizes.
            unsafe { avx2_group_sum(tensor.kind, codes, values, &q2) }
        } else {
            scalar_group_sum(tensor.kind, codes, values, &q2)
        };
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let group_sum = scalar_group_sum(tensor.kind, codes, values, &q2);
        sum += group_sum * factor;
    }
    sum
}

fn matvec(
    data: &[u8],
    tensor: Tensor,
    group_size: usize,
    input: &[f32],
    output: &mut [f32],
    use_avx2: bool,
) {
    debug_assert!(input.len() == tensor.cols && output.len() == tensor.rows);
    debug_assert_eq!(tensor.groups, tensor.rows * (tensor.cols / group_size));
    if output.len() >= 512 && rayon::current_num_threads() > 1 {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, out)| *out = row_dot(data, tensor, group_size, input, row, use_avx2));
    } else {
        for (row, out) in output.iter_mut().enumerate() {
            *out = row_dot(data, tensor, group_size, input, row, use_avx2);
        }
    }
}

fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32, output: &mut [f32]) {
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let factor = 1.0 / (mean_square + epsilon).sqrt();
    for ((out, value), norm_weight) in output.iter_mut().zip(input).zip(weight) {
        *out = *value * factor * *norm_weight;
    }
}

fn rope(values: &mut [f32], heads: usize, head_dim: usize, position: usize, theta: f32) {
    let half = head_dim / 2;
    for head in 0..heads {
        let base = head * head_dim;
        for index in 0..half {
            let frequency = 1.0 / theta.powf((2 * index) as f32 / head_dim as f32);
            let angle = position as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let first = values[base + index];
            let second = values[base + index + half];
            values[base + index] = first * cos - second * sin;
            values[base + index + half] = second * cos + first * sin;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn attention(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    layer: usize,
    position: usize,
    max_seq: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    output: &mut [f32],
    scores: &mut [f32],
) {
    let kv_width = kv_heads * head_dim;
    let repeats = heads / kv_heads;
    let inv_scale = 1.0 / (head_dim as f32).sqrt();
    output.fill(0.0);
    for head in 0..heads {
        let query_start = head * head_dim;
        let kv_start = (head / repeats) * head_dim;
        let mut maximum = f32::NEG_INFINITY;
        for time in 0..=position {
            let cache_start = (layer * max_seq + time) * kv_width + kv_start;
            let mut dot = 0.0;
            for index in 0..head_dim {
                dot += query[query_start + index] * key_cache[cache_start + index];
            }
            let score = dot * inv_scale;
            scores[time] = score;
            maximum = maximum.max(score);
        }
        let mut denominator = 0.0;
        for score in &mut scores[..=position] {
            *score = (*score - maximum).exp();
            denominator += *score;
        }
        for time in 0..=position {
            let probability = scores[time] / denominator;
            let cache_start = (layer * max_seq + time) * kv_width + kv_start;
            for index in 0..head_dim {
                output[query_start + index] += probability * value_cache[cache_start + index];
            }
        }
    }
}

pub fn argmax(values: &[f32]) -> u32 {
    let mut best_index = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }
    best_index as u32
}

pub fn parse_tokens(value: &str) -> io::Result<Vec<u32>> {
    let mut tokens = Vec::new();
    for part in value.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        tokens.push(part.parse::<u32>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid token ID: {part}"),
            )
        })?);
    }
    if tokens.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--tokens must contain at least one token ID",
        ));
    }
    Ok(tokens)
}

pub fn configure_threads(threads: usize) -> io::Result<()> {
    if threads == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--threads must be positive",
        ));
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|index| format!("pickle50-{index}"))
        .build_global()
        .map_err(|error| io::Error::other(format!("cannot configure worker threads: {error}")))
}

pub fn run_prompt<'a>(
    model: &Model,
    state: &'a mut State,
    tokens: &[u32],
) -> io::Result<&'a [f32]> {
    for (position, &token) in tokens.iter().enumerate() {
        model.forward(state, token, position)?;
    }
    Ok(&state.logits)
}

pub fn benchmark_decode(
    model: &Model,
    tokens: &[u32],
    steps: usize,
) -> io::Result<(Vec<u32>, f64, usize)> {
    if steps == 0 || tokens.len() + steps > model.context {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "decode steps must be positive and fit the context window",
        ));
    }
    let mut state = model.state(tokens.len() + steps)?;
    let mut next = argmax(run_prompt(model, &mut state, tokens)?);
    let mut output = Vec::with_capacity(steps);
    let started = Instant::now();
    for step in 0..steps {
        let logits = model.forward(&mut state, next, tokens.len() + step)?;
        next = argmax(logits);
        output.push(next);
    }
    let elapsed = started.elapsed().as_secs_f64();
    Ok((output, elapsed, state.allocated_bytes()))
}

#[cfg(test)]
mod tests {
    use super::half_to_f32;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use super::{avx2_available, avx2_group_sum, scalar_group_sum, Kind};

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn half_conversion_known_values() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert!(half_to_f32(0x7c00).is_infinite());
    }

    #[test]
    fn avx2_group_dot_matches_scalar() {
        if !avx2_available() {
            return;
        }
        let input: Vec<f32> = (0..256)
            .map(|index| ((index as f32 * 0.173).sin() * 3.0) + 0.25)
            .collect();
        let q2_codes: Vec<u8> = (0..64)
            .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
            .collect();
        let q4_codes: Vec<u8> = (0..128)
            .map(|index| (index as u8).wrapping_mul(53).wrapping_add(7))
            .collect();
        let q3_codes: Vec<u8> = (0..96)
            .map(|index| (index as u8).wrapping_mul(41).wrapping_add(3))
            .collect();
        for (kind, codes) in [
            (Kind::Q2, q2_codes),
            (Kind::Q3, q3_codes),
            (Kind::Q4, q4_codes),
        ] {
            let scalar = scalar_group_sum(kind, &codes, &input, &super::Q2);
            // SAFETY: the test returns above when AVX2 is unavailable.
            let vector = unsafe { avx2_group_sum(kind, &codes, &input, &super::Q2) };
            assert!((scalar - vector).abs() < 0.0001, "{scalar} != {vector}");
        }
    }
}
