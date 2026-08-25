//! Self-contained tokenizer dispatch for vocabularies embedded in native
//! models. Supports compact byte-level BPE and the legacy TokenMonster format.
//! TokenMonster's ungreedy branch scoring follows its MIT-licensed Go
//! implementation.

use std::collections::HashMap;
use std::io;

use unicode_normalization::UnicodeNormalization;

const TOKENIZER_MAGIC: &[u8; 4] = b"TMC1";
const BPE_MAGIC: &[u8; 4] = b"BPE1";
const MISSING_U13: usize = 0x1fff;
const MISSING_U16: u16 = 0xffff;
const FLAG_VALUES: [u8; 14] = [1, 3, 4, 5, 16, 17, 128, 131, 132, 133, 136, 140, 152, 165];
const UNICODE13_CATEGORIES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/unicode13_categories.bin"));
const NO_CAP_DELETE: u8 = 0x7f;
const NO_CAP_SUBSTITUTE: u8 = 0x14;
const INVALID_SCORE: i32 = -1_000_000;

#[derive(Clone)]
struct Info {
    token: Vec<u8>,
    flag: u8,
    words: u8,
    alt1: Option<usize>,
    alt2: Option<usize>,
    id: u32,
}

#[derive(Clone, Copy)]
struct Candidate {
    score: i32,
    first_id: u32,
    first_len: usize,
    next_index: usize,
    next_len: usize,
    delete: bool,
}

pub struct TokenMonsterTokenizer {
    capcode: u8,
    charset: u8,
    normalization: u8,
    unk: Option<u32>,
    delete_token: Option<u32>,
    max_token_len: usize,
    info: Vec<Info>,
    begin_byte: [u8; 256],
    reverse: Vec<Vec<u8>>,
    buckets: Vec<Vec<usize>>,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn read_u16(data: &[u8], cursor: &mut usize) -> io::Result<u16> {
    let raw = data
        .get(*cursor..*cursor + 2)
        .ok_or_else(|| invalid("truncated TokenMonster vocabulary"))?;
    *cursor += 2;
    Ok(u16::from_le_bytes([raw[0], raw[1]]))
}

fn optional_u16(value: u16) -> Option<u32> {
    (value != MISSING_U16).then_some(value as u32)
}

fn unicode13_class(ch: char) -> u8 {
    let codepoint = ch as u32;
    let mut low = 0;
    let mut high = UNICODE13_CATEGORIES.len() / 7;
    while low < high {
        let middle = (low + high) / 2;
        let offset = middle * 7;
        let start = UNICODE13_CATEGORIES[offset] as u32
            | ((UNICODE13_CATEGORIES[offset + 1] as u32) << 8)
            | ((UNICODE13_CATEGORIES[offset + 2] as u32) << 16);
        let end = UNICODE13_CATEGORIES[offset + 3] as u32
            | ((UNICODE13_CATEGORIES[offset + 4] as u32) << 8)
            | ((UNICODE13_CATEGORIES[offset + 5] as u32) << 16);
        if codepoint < start {
            high = middle;
        } else if codepoint > end {
            low = middle + 1;
        } else {
            return UNICODE13_CATEGORIES[offset + 6];
        }
    }
    0
}

fn normalize_nfd13(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut assigned_run = String::new();
    for ch in text.chars() {
        if unicode13_class(ch) != 0 {
            assigned_run.push(ch);
        } else {
            output.extend(assigned_run.nfd());
            assigned_run.clear();
            output.push(ch);
        }
    }
    output.extend(assigned_run.nfd());
    output
}

fn is_letter(ch: char) -> bool {
    unicode13_class(ch) == 1
}

fn is_number(ch: char) -> bool {
    unicode13_class(ch) == 2
}

fn is_modifier(ch: char) -> bool {
    unicode13_class(ch) == 3
}

fn no_capcode_encode(text: &str) -> Vec<u8> {
    let mut output = Vec::with_capacity(text.len() + text.len() / 2 + 8);
    let mut previous = '\0';
    let mut previous2 = '\0';
    for ch in text.chars() {
        if is_letter(ch) {
            if !(previous == ' '
                || is_letter(previous)
                || (is_letter(previous2) && (previous == '\'' || previous == '’'))
                || is_modifier(previous))
            {
                output.extend_from_slice(&[NO_CAP_DELETE, b' ']);
            }
        } else if is_number(ch) && !(previous == ' ' || is_number(previous)) {
            output.extend_from_slice(&[NO_CAP_DELETE, b' ']);
        }

        if ch == NO_CAP_DELETE as char {
            output.push(NO_CAP_SUBSTITUTE);
        } else {
            let mut encoded = [0_u8; 4];
            output.extend_from_slice(ch.encode_utf8(&mut encoded).as_bytes());
        }
        previous2 = previous;
        previous = ch;
    }
    output
}

fn no_capcode_decode(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len());
    let mut index = 0;
    while index < input.len() {
        if input[index] == NO_CAP_DELETE {
            index += 2;
        } else {
            output.push(input[index]);
            index += 1;
        }
    }
    output
}

fn max_zero(value: i32) -> i32 {
    value.max(0)
}

impl TokenMonsterTokenizer {
    pub fn load(data: &[u8], expected_vocab: usize) -> io::Result<Self> {
        if data.len() < 18 || data.get(..4) != Some(TOKENIZER_MAGIC) {
            return Err(invalid("truncated TokenMonster vocabulary header"));
        }
        let capcode = data[4];
        let charset = data[5];
        let normalization = data[6];
        let max_token_len = data[7] as usize;
        if capcode > 2 || charset > 2 {
            return Err(invalid("invalid TokenMonster vocabulary header"));
        }
        if charset != 1 || capcode != 1 || normalization != 1 {
            return Err(invalid(
                "this model runtime requires UTF-8, NFD, no-capcode TokenMonster data",
            ));
        }

        let mut cursor = 8;
        let vocab_size = read_u16(data, &mut cursor)? as usize;
        let reverse_count = read_u16(data, &mut cursor)? as usize;
        let info_count = read_u16(data, &mut cursor)? as usize;
        let unk = optional_u16(read_u16(data, &mut cursor)?);
        let delete_token = optional_u16(read_u16(data, &mut cursor)?);
        if vocab_size != expected_vocab || reverse_count != expected_vocab || max_token_len == 0 {
            return Err(invalid(
                "TokenMonster vocabulary dimensions do not match model",
            ));
        }

        let mut info: Vec<Info> = Vec::with_capacity(info_count);
        let mut reverse = vec![Vec::new(); reverse_count];
        let mut next_new_id = 0_u32;
        for index in 0..info_count {
            let packed_bytes = data
                .get(cursor..cursor + 5)
                .ok_or_else(|| invalid("truncated TokenMonster token metadata"))?;
            let packed = packed_bytes
                .iter()
                .enumerate()
                .fold(0_u64, |value, (shift, byte)| {
                    value | ((*byte as u64) << (shift * 8))
                });
            cursor += 5;
            let length = (packed & 0x3f) as usize;
            if length == 0 || length > max_token_len {
                return Err(invalid("invalid TokenMonster token length"));
            }
            let flag_index = ((packed >> 6) & 0xf) as usize;
            let flag = *FLAG_VALUES
                .get(flag_index)
                .ok_or_else(|| invalid("invalid compact TokenMonster flag"))?;
            let words = ((packed >> 10) & 0x7) as u8;
            let alt1_raw = ((packed >> 13) & 0x1fff) as usize;
            let alt2_raw = ((packed >> 26) & 0x1fff) as usize;
            let is_new_id = ((packed >> 39) & 1) != 0;
            let prefix = *data
                .get(cursor)
                .ok_or_else(|| invalid("truncated TokenMonster token prefix"))?
                as usize;
            cursor += 1;
            let id = if is_new_id {
                let id = next_new_id;
                next_new_id += 1;
                id
            } else {
                read_u16(data, &mut cursor)? as u32
            };
            let previous = info
                .last()
                .map(|entry| entry.token.as_slice())
                .unwrap_or(&[]);
            if prefix > previous.len() || prefix > length {
                return Err(invalid("invalid TokenMonster token prefix"));
            }
            let suffix_length = length - prefix;
            let end = cursor
                .checked_add(suffix_length)
                .ok_or_else(|| invalid("TokenMonster token length overflow"))?;
            let suffix = data
                .get(cursor..end)
                .ok_or_else(|| invalid("truncated TokenMonster token"))?;
            let mut token = Vec::with_capacity(length);
            token.extend_from_slice(&previous[..prefix]);
            token.extend_from_slice(suffix);
            cursor = end;
            let alt1 = (alt1_raw != MISSING_U13).then_some(alt1_raw);
            let alt2 = (alt2_raw != MISSING_U13).then_some(alt2_raw);
            if alt1.is_some_and(|value| value >= index)
                || alt2.is_some_and(|value| value >= index)
                || id as usize >= reverse_count
            {
                return Err(invalid("invalid TokenMonster token index"));
            }
            reverse[id as usize] = token.clone();
            info.push(Info {
                token,
                flag,
                words,
                alt1,
                alt2,
                id,
            });
        }

        let begin_slice = data
            .get(cursor..cursor + 256)
            .ok_or_else(|| invalid("truncated TokenMonster begin-byte table"))?;
        let mut begin_byte = [0_u8; 256];
        begin_byte.copy_from_slice(begin_slice);
        cursor += 256;
        if cursor != data.len() || next_new_id as usize != reverse_count {
            return Err(invalid("invalid compact TokenMonster vocabulary length"));
        }

        let mut buckets = vec![Vec::new(); 256];
        for (index, entry) in info.iter().enumerate() {
            buckets[entry.token[0] as usize].push(index);
        }
        for bucket in &mut buckets {
            bucket.sort_by_key(|&index| std::cmp::Reverse(info[index].token.len()));
        }

        Ok(Self {
            capcode,
            charset,
            normalization,
            unk,
            delete_token,
            max_token_len,
            info,
            begin_byte,
            reverse,
            buckets,
        })
    }

    fn longest(&self, data: &[u8]) -> Option<(usize, usize)> {
        let first = *data.first()?;
        let maximum = data.len().min(self.max_token_len);
        for &index in &self.buckets[first as usize] {
            let token = &self.info[index].token;
            if token.len() <= maximum && data.starts_with(token) {
                return Some((index, token.len()));
            }
        }
        None
    }

    fn begin_at(&self, data: &[u8], index: usize) -> u8 {
        self.begin_byte[data.get(index).copied().unwrap_or(0) as usize]
    }

    fn score(
        &self,
        first: &Info,
        second: &Info,
        branch_len: usize,
        forward_delete: i32,
        next_byte: u8,
        inserted_delete: bool,
        original_len: usize,
        alternative: bool,
    ) -> i32 {
        let words = first.words as i32 - forward_delete;
        let second_words = second.words as i32;
        let mut score = branch_len as i32
            + (first.flag >> 7) as i32
            + (second.flag >> 7) as i32
            + max_zero(words - 1)
            + max_zero(second_words - 1)
            + if inserted_delete {
                0
            } else {
                ((second.flag >> 2) & 1) as i32
            }
            + ((next_byte >> 2) & 1) as i32
            + (words + second_words + (next_byte >> 3) as i32) * 100;

        score -= if inserted_delete {
            ((first.flag & 1) as i32 * 103)
                + (((first.flag >> 3) & 1 & (second.flag >> 4)) as i32 * 100)
                + ((second.flag & 1 & next_byte) as i32 * 3)
                + 1
        } else {
            ((first.flag & 1 & (second.flag >> 1)) as i32 * 103)
                + (((first.flag >> 3) & 1 & (second.flag >> 4)) as i32 * 100)
                + ((second.flag & 1 & next_byte) as i32 * 3)
        };
        if alternative {
            if branch_len < original_len {
                score -= 100;
            } else if branch_len == original_len {
                score -= 10_000;
            }
        }
        score
    }

    fn normal_candidate(
        &self,
        data: &[u8],
        i: usize,
        first_index: usize,
        first_len: usize,
        forward_delete: i32,
        original_len: usize,
        alternative: bool,
    ) -> Option<Candidate> {
        let next_at = i + first_len;
        let (next_index, next_len) = self.longest(data.get(next_at..)?)?;
        let next_byte = self.begin_at(data, next_at + next_len);
        let score = self.score(
            &self.info[first_index],
            &self.info[next_index],
            first_len + next_len,
            forward_delete,
            next_byte,
            false,
            original_len,
            alternative,
        );
        Some(Candidate {
            score,
            first_id: self.info[first_index].id,
            first_len,
            next_index,
            next_len,
            delete: false,
        })
    }

    fn delete_candidate(
        &self,
        data: &[u8],
        i: usize,
        first_index: usize,
        first_len: usize,
        forward_delete: i32,
        normal: Candidate,
        original_len: usize,
        alternative: bool,
    ) -> Option<Candidate> {
        self.delete_token?;
        let normal_second = &self.info[normal.next_index];
        let normal_next_byte = self.begin_at(data, i + first_len + normal.next_len);
        if normal_second.flag & 2 == 0 || normal_next_byte != 1 || normal_second.words != 0 {
            return None;
        }

        let remaining = data.get(i + first_len..)?;
        let copied = remaining
            .len()
            .min(self.max_token_len - self.charset as usize);
        let mut with_space = Vec::with_capacity(copied + self.charset as usize);
        with_space.extend(std::iter::repeat_n(b' ', self.charset as usize));
        with_space.extend_from_slice(&remaining[..copied]);
        let (next_index, length_with_space) = self.longest(&with_space)?;
        if length_with_space <= normal.next_len + self.charset as usize {
            return None;
        }
        let next_len = length_with_space - self.charset as usize;
        let next_byte = self.begin_at(data, i + first_len + next_len);
        let score = self.score(
            &self.info[first_index],
            &self.info[next_index],
            first_len + next_len,
            forward_delete,
            next_byte,
            true,
            original_len,
            alternative,
        );
        Some(Candidate {
            score,
            first_id: self.info[first_index].id,
            first_len,
            next_index,
            next_len,
            delete: true,
        })
    }

    fn tokenize_normalized(&self, data: &[u8]) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(data.len() / 2 + 4);
        let mut i = 0;
        let mut forward_delete = 0_i32;
        while i < data.len() {
            let Some((mut index, mut length)) = self.longest(&data[i..]) else {
                if let Some(unk) = self.unk {
                    tokens.push(unk);
                }
                i += 1;
                forward_delete = 0;
                continue;
            };

            loop {
                let original = &self.info[index];
                let after = i + length;
                if after < data.len()
                    && (original.flag & 32 == 0 || self.begin_at(data, after) != 12)
                {
                    let mut normal = [None, None, None];
                    let mut deleted = [None, None, None];
                    normal[0] = self.normal_candidate(
                        data,
                        i,
                        index,
                        length,
                        forward_delete,
                        length,
                        false,
                    );
                    if let Some(first) = normal[0] {
                        deleted[0] = self.delete_candidate(
                            data,
                            i,
                            index,
                            length,
                            forward_delete,
                            first,
                            length,
                            false,
                        );
                    }

                    for (slot, alternative_index) in
                        [original.alt1, original.alt2].into_iter().enumerate()
                    {
                        if let Some(first_index) = alternative_index {
                            let raw_len = self.info[first_index].token.len();
                            if raw_len >= forward_delete as usize {
                                let first_len = raw_len - forward_delete as usize;
                                normal[slot + 1] = self.normal_candidate(
                                    data,
                                    i,
                                    first_index,
                                    first_len,
                                    forward_delete,
                                    length,
                                    true,
                                );
                                if let Some(first) = normal[slot + 1] {
                                    deleted[slot + 1] = self.delete_candidate(
                                        data,
                                        i,
                                        first_index,
                                        first_len,
                                        forward_delete,
                                        first,
                                        length,
                                        true,
                                    );
                                }
                            }
                        }
                    }

                    let max_score = normal
                        .iter()
                        .chain(deleted.iter())
                        .flatten()
                        .map(|candidate| candidate.score)
                        .max()
                        .unwrap_or(INVALID_SCORE);
                    let selected = normal
                        .into_iter()
                        .chain(deleted)
                        .flatten()
                        .find(|candidate| candidate.score == max_score);
                    if let Some(candidate) = selected {
                        tokens.push(candidate.first_id);
                        if candidate.delete {
                            tokens.push(self.delete_token.unwrap());
                        }
                        i += candidate.first_len;
                        index = candidate.next_index;
                        length = candidate.next_len;
                        forward_delete = i32::from(candidate.delete);
                        continue;
                    }
                }

                tokens.push(original.id);
                i += length;
                forward_delete = 0;
                break;
            }
        }
        tokens
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let normalized = if self.normalization == 1 {
            normalize_nfd13(text)
        } else {
            text.to_owned()
        };
        let bytes = if self.capcode == 1 {
            no_capcode_encode(&normalized)
        } else {
            normalized.into_bytes()
        };
        self.tokenize_normalized(&bytes)
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut raw = Vec::new();
        for &token in tokens {
            if let Some(piece) = self.reverse.get(token as usize) {
                raw.extend_from_slice(piece);
            }
        }
        let decoded = if self.capcode == 1 {
            no_capcode_decode(&raw)
        } else {
            raw
        };
        String::from_utf8_lossy(&decoded).into_owned()
    }
}

pub struct ByteBpeTokenizer {
    byte_ids: [u16; 256],
    reverse: Vec<Vec<u8>>,
    merges: HashMap<(u16, u16), (u32, u16)>,
    specials: Vec<(Vec<u8>, u16)>,
}

impl ByteBpeTokenizer {
    fn load(data: &[u8], expected_vocab: usize) -> io::Result<Self> {
        if data.len() < 14 || data.get(..4) != Some(BPE_MAGIC) {
            return Err(invalid("truncated byte-level BPE header"));
        }
        let mut cursor = 4;
        let vocab_size = read_u32_bpe(data, &mut cursor)? as usize;
        let merge_count = read_u32_bpe(data, &mut cursor)? as usize;
        let special_count = read_u16(data, &mut cursor)? as usize;
        if vocab_size != expected_vocab || vocab_size > u16::MAX as usize {
            return Err(invalid("byte-level BPE dimensions do not match model"));
        }
        let mut byte_ids = [0_u16; 256];
        for value in &mut byte_ids {
            *value = read_u16(data, &mut cursor)?;
            if *value as usize >= vocab_size {
                return Err(invalid("byte-level BPE byte ID is outside vocabulary"));
            }
        }
        let mut reverse = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let length = read_u16(data, &mut cursor)? as usize;
            let end = cursor
                .checked_add(length)
                .ok_or_else(|| invalid("byte-level BPE token length overflow"))?;
            reverse.push(
                data.get(cursor..end)
                    .ok_or_else(|| invalid("truncated byte-level BPE token"))?
                    .to_vec(),
            );
            cursor = end;
        }
        let mut merges = HashMap::with_capacity(merge_count);
        for rank in 0..merge_count {
            let left = read_u16(data, &mut cursor)?;
            let right = read_u16(data, &mut cursor)?;
            let merged = read_u16(data, &mut cursor)?;
            if [left, right, merged]
                .into_iter()
                .any(|value| value as usize >= vocab_size)
                || merges
                    .insert((left, right), (rank as u32, merged))
                    .is_some()
            {
                return Err(invalid("invalid or duplicate byte-level BPE merge"));
            }
        }
        let mut specials = Vec::with_capacity(special_count);
        for _ in 0..special_count {
            let token_id = read_u16(data, &mut cursor)?;
            let length = read_u16(data, &mut cursor)? as usize;
            let end = cursor
                .checked_add(length)
                .ok_or_else(|| invalid("byte-level BPE special-token length overflow"))?;
            let raw = data
                .get(cursor..end)
                .ok_or_else(|| invalid("truncated byte-level BPE special token"))?
                .to_vec();
            if raw.is_empty() || token_id as usize >= vocab_size {
                return Err(invalid("invalid byte-level BPE special token"));
            }
            specials.push((raw, token_id));
            cursor = end;
        }
        if cursor != data.len() {
            return Err(invalid("trailing bytes in byte-level BPE tokenizer"));
        }
        // Longest first gives deterministic matching if special strings overlap.
        specials.sort_by(|left, right| right.0.len().cmp(&left.0.len()));
        Ok(Self {
            byte_ids,
            reverse,
            merges,
            specials,
        })
    }

    fn bpe_piece(&self, piece: &str, output: &mut Vec<u32>) {
        let mut tokens: Vec<u16> = piece
            .as_bytes()
            .iter()
            .map(|byte| self.byte_ids[*byte as usize])
            .collect();
        while tokens.len() > 1 {
            let selected = tokens
                .windows(2)
                .enumerate()
                .filter_map(|(index, pair)| {
                    self.merges
                        .get(&(pair[0], pair[1]))
                        .map(|&(rank, merged)| (rank, index, merged))
                })
                .min_by_key(|&(rank, index, _)| (rank, index));
            let Some((_, index, merged)) = selected else {
                break;
            };
            tokens[index] = merged;
            tokens.remove(index + 1);
        }
        output.extend(tokens.into_iter().map(u32::from));
    }

    fn encode_ordinary(&self, text: &str, output: &mut Vec<u32>) {
        let mut cursor = 0;
        while cursor < text.len() {
            let remaining = &text[cursor..];
            let mut end = cursor;

            // GPT-2 contractions have priority over punctuation.
            if remaining.starts_with('\'') {
                let lowercase = remaining.to_ascii_lowercase();
                if let Some(suffix) = ["'re", "'ve", "'ll", "'s", "'t", "'m", "'d"]
                    .into_iter()
                    .find(|suffix| lowercase.starts_with(suffix))
                {
                    end = cursor + suffix.len();
                }
            }

            if end == cursor {
                let mut scan = cursor;
                if text[scan..].starts_with(' ') {
                    scan += 1;
                }
                if let Some(first) = text[scan..].chars().next() {
                    let class = char_class(first);
                    if class != 0 {
                        end = scan + first.len_utf8();
                        while let Some(ch) = text[end..].chars().next() {
                            if char_class(ch) != class {
                                break;
                            }
                            end += ch.len_utf8();
                        }
                    }
                }
            }

            if end == cursor {
                let first = remaining.chars().next().unwrap();
                if first.is_whitespace() {
                    let mut whitespace_end = cursor + first.len_utf8();
                    let mut starts = vec![cursor, whitespace_end];
                    while let Some(ch) = text[whitespace_end..].chars().next() {
                        if !ch.is_whitespace() {
                            break;
                        }
                        whitespace_end += ch.len_utf8();
                        starts.push(whitespace_end);
                    }
                    // `\s+(?!\S)` consumes all trailing whitespace, or all but
                    // the final character when a non-space follows.
                    end = if whitespace_end == text.len() || starts.len() == 2 {
                        whitespace_end
                    } else {
                        starts[starts.len() - 2]
                    };
                } else {
                    end = cursor + first.len_utf8();
                    while let Some(ch) = text[end..].chars().next() {
                        if ch.is_whitespace() || is_letter(ch) || is_number(ch) {
                            break;
                        }
                        end += ch.len_utf8();
                    }
                }
            }

            self.bpe_piece(&text[cursor..end], output);
            cursor = end;
        }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let raw = text.as_bytes();
        let mut output = Vec::with_capacity(raw.len() / 3 + 4);
        let mut cursor = 0;
        while cursor < raw.len() {
            let next = self
                .specials
                .iter()
                .filter_map(|(special, token_id)| {
                    raw[cursor..]
                        .windows(special.len())
                        .position(|window| window == special)
                        .map(|offset| (offset, special.len(), *token_id))
                })
                .min_by_key(|&(offset, length, _)| (offset, usize::MAX - length));
            if let Some((offset, length, token_id)) = next {
                let start = cursor + offset;
                self.encode_ordinary(&text[cursor..start], &mut output);
                output.push(token_id as u32);
                cursor = start + length;
            } else {
                self.encode_ordinary(&text[cursor..], &mut output);
                break;
            }
        }
        output
    }

    fn decode(&self, tokens: &[u32]) -> String {
        let mut raw = Vec::new();
        for &token in tokens {
            if let Some(piece) = self.reverse.get(token as usize) {
                raw.extend_from_slice(piece);
            }
        }
        String::from_utf8_lossy(&raw).into_owned()
    }
}

fn read_u32_bpe(data: &[u8], cursor: &mut usize) -> io::Result<u32> {
    let raw = data
        .get(*cursor..*cursor + 4)
        .ok_or_else(|| invalid("truncated byte-level BPE tokenizer"))?;
    *cursor += 4;
    Ok(u32::from_le_bytes(raw.try_into().unwrap()))
}

fn char_class(ch: char) -> u8 {
    if is_letter(ch) {
        1
    } else if is_number(ch) {
        2
    } else if !ch.is_whitespace() {
        3
    } else {
        0
    }
}

pub enum Tokenizer {
    TokenMonster(TokenMonsterTokenizer),
    ByteBpe(ByteBpeTokenizer),
}

impl Tokenizer {
    pub fn load(data: &[u8], expected_vocab: usize) -> io::Result<Self> {
        match data.get(..4) {
            Some(magic) if magic == TOKENIZER_MAGIC => Ok(Self::TokenMonster(
                TokenMonsterTokenizer::load(data, expected_vocab)?,
            )),
            Some(magic) if magic == BPE_MAGIC => {
                Ok(Self::ByteBpe(ByteBpeTokenizer::load(data, expected_vocab)?))
            }
            _ => Err(invalid("unsupported native tokenizer format")),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Self::TokenMonster(tokenizer) => tokenizer.encode(text),
            Self::ByteBpe(tokenizer) => tokenizer.encode(text),
        }
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        match self {
            Self::TokenMonster(tokenizer) => tokenizer.decode(tokens),
            Self::ByteBpe(tokenizer) => tokenizer.decode(tokens),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::TokenMonster(_) => "TokenMonster",
            Self::ByteBpe(_) => "byte-level-bpe",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_capcode_boundaries_round_trip() {
        let source = "hello.World 42+7 café";
        let normalized = source.nfd().collect::<String>();
        let encoded = no_capcode_encode(&normalized);
        assert_eq!(
            String::from_utf8(no_capcode_decode(&encoded)).unwrap(),
            normalized
        );
        assert!(encoded.contains(&NO_CAP_DELETE));
    }
}
