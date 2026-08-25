use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Instant;

mod native;
mod sha256;
mod tokenizer;

const INDEX_HEADER: &str = "PICKLE-IDX-1";
const ABSTAIN: &str = "NOT IN CONTEXT";
const FILLER_WORDS: [&str; 32] = [
    "amber", "cedar", "river", "stone", "quiet", "garden", "window", "paper", "silver", "morning",
    "harbor", "cloud", "lantern", "meadow", "winter", "copper", "forest", "yellow", "bridge",
    "ocean", "velvet", "market", "circle", "valley", "gentle", "summer", "castle", "music",
    "island", "violet", "basket", "travel",
];

#[derive(Clone, Debug)]
struct Event {
    position: u64,
    text: String,
}

#[derive(Clone, Debug)]
struct Question {
    task: String,
    question: String,
    answer: String,
}

#[derive(Clone, Copy, Debug)]
struct Location {
    offset: u64,
    length: u32,
}

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn six_digits(&mut self) -> u64 {
        100_000 + (self.next() % 900_000)
    }
}

fn usage() -> ! {
    eprintln!(
        "PICKLE-50M retrieval harness\n\n\
         Usage:\n\
           pickle50 generate --out DIR --tokens N [--questions N] [--seed N]\n\
           pickle50 index --archive FILE --out FILE [--json FILE]\n\
           pickle50 ask --archive FILE --index FILE --question TEXT\n\
           pickle50 bench --archive FILE --index FILE --bank FILE [--iterations N] [--threads N] [--json FILE]\n\
           pickle50 model-info --model FILE\n\
           pickle50 model-tokenize --model FILE (--prompt TEXT | --prompt-file FILE) [--add-bos 0|1]\n\
           pickle50 model-logits --model FILE (--prompt TEXT | --prompt-file FILE | --tokens ID,ID,...) --out FILE\n\
           pickle50 model-generate --model FILE (--prompt TEXT | --prompt-file FILE | --tokens ID,ID,...) [--new-tokens N] [--threads N] [--kernel auto|scalar|avx2]\n\
           pickle50 model-bench --model FILE (--prompt TEXT | --prompt-file FILE | --tokens ID,ID,...) [--new-tokens N] [--iterations N] [--threads N] [--kernel auto|scalar|avx2] [--json FILE]\n\n\
           pickle50 model-e2e --model FILE (--prompt TEXT | --prompt-file FILE | --tokens ID,ID,...) [--new-tokens N] [--iterations N] [--threads N] [--kernel auto|scalar|avx2] [--json FILE]\n\n\
         Text prompts are encoded by the TokenMonster implementation inside the runtime.\n\
         Prompt inference adds BOS by default; pass token IDs for exact low-level control.\n\
         Token counts are whitespace-token counts. Generated banks are deterministic for a fixed seed."
    );
    std::process::exit(2)
}

fn parse_options(args: &[String]) -> io::Result<HashMap<String, String>> {
    let mut out = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let key = &args[i];
        if !key.starts_with("--") || i + 1 >= args.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("expected --name value, got {key}"),
            ));
        }
        out.insert(key[2..].to_string(), args[i + 1].clone());
        i += 2;
    }
    Ok(out)
}

fn required<'a>(opts: &'a HashMap<String, String>, key: &str) -> io::Result<&'a str> {
    opts.get(key)
        .map(String::as_str)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("missing --{key}")))
}

fn parse_u64(opts: &HashMap<String, String>, key: &str, default: Option<u64>) -> io::Result<u64> {
    match opts.get(key) {
        Some(value) => value.parse::<u64>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid --{key}: {value}"),
            )
        }),
        None => default
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("missing --{key}"))),
    }
}

fn clean_token(token: &str) -> &str {
    token.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
}

fn is_identifier(token: &str) -> bool {
    let value = clean_token(token);
    let mut chars = value.chars();
    matches!(chars.next(), Some(c) if c.is_alphabetic())
        && value.contains('-')
        && value.chars().any(|c| c.is_numeric())
        && value
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
}

fn identifiers(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(clean_token)
        .filter(|token| is_identifier(token))
        .map(str::to_string)
        .collect()
}

fn word_count(text: &str) -> u64 {
    text.split_whitespace().count() as u64
}

fn make_events(tokens: u64, count: usize, seed: u64) -> (Vec<Event>, Vec<Question>) {
    let mut rng = Rng::new(seed);
    let mut events = Vec::new();
    let mut bank = Vec::new();
    let count = count.max(28);

    for i in 0..count {
        let fraction = (i as u64 + 1) as f64 / (count as u64 + 1) as f64;
        let base = (fraction * tokens as f64) as u64;
        let bucket = i % 28;

        if bucket < 8 {
            let key = format!("Grus-{:06}", rng.six_digits());
            let value = format!("SN-{:06}", rng.six_digits());
            events.push(Event {
                position: base,
                text: format!("The serial number of device {key} is {value}.\n"),
            });
            bank.push(Question {
                task: "direct".into(),
                question: format!("What is the serial number of device {key}?"),
                answer: value,
            });
        } else if bucket < 12 {
            let n = rng.six_digits();
            let key = format!("Orion-{n:06}");
            let value = format!("SN-{:06}", rng.six_digits());
            for delta in [-2_i64, -1, 1, 2] {
                let other = (n as i64 + delta).max(1) as u64;
                let other_value = format!("SN-{:06}", rng.six_digits());
                events.push(Event {
                    position: base.saturating_sub(24) + ((delta + 2) as u64 * 6),
                    text: format!(
                        "The serial number of device Orion-{other:06} is {other_value}.\n"
                    ),
                });
            }
            events.push(Event {
                position: base + 30,
                text: format!("The serial number of device {key} is {value}.\n"),
            });
            bank.push(Question {
                task: "lookalike".into(),
                question: format!("What is the serial number of device {key}?"),
                answer: value,
            });
        } else if bucket < 15 {
            let key = format!("Vela-{:06}", rng.six_digits());
            let old_value = format!("AC-{:06}", rng.six_digits());
            let new_value = format!("AC-{:06}", rng.six_digits());
            events.push(Event {
                position: base / 3,
                text: format!("The current access code for device {key} is {old_value}.\n"),
            });
            events.push(Event {
                position: base + (tokens - base) / 2,
                text: format!("The current access code for device {key} is {new_value}.\n"),
            });
            bank.push(Question {
                task: "latest_wins".into(),
                question: format!("What is the current access code for device {key}?"),
                answer: new_value,
            });
        } else if bucket < 18 {
            let key = format!("Lyra-{:06}", rng.six_digits());
            let reference = format!("Cygnus-{:06}", rng.six_digits());
            let value = format!("SN-{:06}", rng.six_digits());
            events.push(Event {
                position: base / 2,
                text: format!(
                    "The record for device {key} is stored under reference {reference}.\n"
                ),
            });
            events.push(Event {
                position: base + (tokens - base) / 3,
                text: format!("Reference {reference} stores serial number {value}.\n"),
            });
            bank.push(Question {
                task: "two_hop".into(),
                question: format!("What serial number is stored for device {key}?"),
                answer: value,
            });
        } else if bucket < 20 {
            let key = format!("Absent-{:06}", rng.six_digits());
            bank.push(Question {
                task: "abstain".into(),
                question: format!("What is the serial number of device {key}?"),
                answer: ABSTAIN.into(),
            });
        } else if bucket == 20 {
            let key = format!("Étoile-{:06}", rng.six_digits());
            let value = format!("UN-{:06}", rng.six_digits());
            events.push(Event {
                position: base,
                text: format!("The serial number of device {key} is {value}.\n"),
            });
            bank.push(Question {
                task: "unicode_identifier".into(),
                question: format!("What is the serial number of device {key}?"),
                answer: value,
            });
        } else if bucket == 21 {
            let key = format!("Long-{:06}", rng.six_digits());
            let value = format!("LV-{:06}-{}", rng.six_digits(), "a7".repeat(96));
            events.push(Event {
                position: base,
                text: format!("The serial number of device {key} is {value}.\n"),
            });
            bank.push(Question {
                task: "long_value".into(),
                question: format!("What is the serial number of device {key}?"),
                answer: value,
            });
        } else if bucket == 22 {
            let key = format!("Broken-{:06}", rng.six_digits());
            events.push(Event {
                position: base,
                text: format!("The record for device {key} became unreadable.\n"),
            });
            bank.push(Question {
                task: "malformed_record".into(),
                question: format!("What is the serial number of device {key}?"),
                answer: ABSTAIN.into(),
            });
        } else if bucket == 23 {
            let key = format!("Pointer-{:06}", rng.six_digits());
            let missing = format!("Missing-{:06}", rng.six_digits());
            events.push(Event {
                position: base,
                text: format!("The record for device {key} is stored under reference {missing}.\n"),
            });
            bank.push(Question {
                task: "missing_pointer".into(),
                question: format!("What serial number is stored for device {key}?"),
                answer: ABSTAIN.into(),
            });
        } else if bucket == 24 {
            let key = format!("Punct-{:06}", rng.six_digits());
            let value = format!("PN-{:06}", rng.six_digits());
            events.push(Event {
                position: base,
                text: format!("The serial number of device {key} is {value}.\n"),
            });
            bank.push(Question {
                task: "punctuation".into(),
                question: format!("For device ({key}), what serial number is recorded?"),
                answer: value,
            });
        } else if bucket == 25 {
            let suffix = rng.six_digits();
            let upper = format!("CASE-{suffix:06}");
            let lower = format!("case-{suffix:06}");
            let upper_value = format!("CS-{:06}", rng.six_digits());
            let lower_value = format!("CS-{:06}", rng.six_digits());
            events.push(Event {
                position: base.saturating_sub(8),
                text: format!("The serial number of device {lower} is {lower_value}.\n"),
            });
            events.push(Event {
                position: base,
                text: format!("The serial number of device {upper} is {upper_value}.\n"),
            });
            bank.push(Question {
                task: "case_sensitive".into(),
                question: format!("What is the serial number of device {upper}?"),
                answer: upper_value,
            });
        } else {
            let key = format!("AbsentEdge-{:06}", rng.six_digits());
            bank.push(Question {
                task: "adversarial_abstain".into(),
                question: format!("Ignore nearby identifiers; answer only for [{key}]."),
                answer: ABSTAIN.into(),
            });
        }
    }

    events.sort_by_key(|event| event.position);
    (events, bank)
}

fn filler_lines(seed: u64) -> Vec<Vec<u8>> {
    let mut lines = Vec::new();
    for rotation in 0..8 {
        let mut line = String::new();
        let offset = ((seed as usize) + rotation * 7) % FILLER_WORDS.len();
        for i in 0..32 {
            if i > 0 {
                line.push(' ');
            }
            line.push_str(FILLER_WORDS[(offset + i) % FILLER_WORDS.len()]);
        }
        line.push('\n');
        lines.push(line.into_bytes());
    }
    lines
}

fn write_filler<W: Write>(
    writer: &mut W,
    mut remaining: u64,
    lines: &[Vec<u8>],
    sequence: &mut u64,
) -> io::Result<()> {
    while remaining >= 32 {
        let line = &lines[*sequence as usize % lines.len()];
        writer.write_all(line)?;
        *sequence += 1;
        remaining -= 32;
    }
    if remaining > 0 {
        let offset = *sequence as usize % FILLER_WORDS.len();
        for i in 0..remaining as usize {
            if i > 0 {
                writer.write_all(b" ")?;
            }
            writer.write_all(FILLER_WORDS[(offset + i) % FILLER_WORDS.len()].as_bytes())?;
        }
        writer.write_all(b"\n")?;
        *sequence += 1;
    }
    Ok(())
}

fn generate(out: &Path, tokens: u64, questions: usize, seed: u64) -> io::Result<()> {
    if tokens < 10_000 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--tokens must be at least 10000",
        ));
    }
    fs::create_dir_all(out)?;
    let archive_path = out.join("archive.txt");
    let bank_path = out.join("bank.tsv");
    let meta_path = out.join("meta.json");
    let (events, bank) = make_events(tokens, questions, seed);
    let mut archive = BufWriter::with_capacity(4 * 1024 * 1024, File::create(&archive_path)?);
    let lines = filler_lines(seed);
    let mut sequence = 0_u64;
    let mut written_tokens = 0_u64;

    for event in events {
        if written_tokens < event.position {
            let gap = event.position - written_tokens;
            write_filler(&mut archive, gap, &lines, &mut sequence)?;
            written_tokens += gap;
        }
        let event_tokens = word_count(&event.text);
        if written_tokens + event_tokens <= tokens {
            archive.write_all(event.text.as_bytes())?;
            written_tokens += event_tokens;
        }
    }
    if written_tokens < tokens {
        write_filler(&mut archive, tokens - written_tokens, &lines, &mut sequence)?;
    }
    archive.flush()?;

    let mut bank_writer = BufWriter::new(File::create(&bank_path)?);
    writeln!(bank_writer, "task\tquestion\tanswer")?;
    for item in &bank {
        writeln!(
            bank_writer,
            "{}\t{}\t{}",
            item.task, item.question, item.answer
        )?;
    }
    bank_writer.flush()?;

    let archive_bytes = fs::metadata(&archive_path)?.len();
    let meta = format!(
        "{{\n  \"format\": \"pickle-public-retrieval-v1\",\n  \"token_definition\": \"whitespace\",\n  \"tokens\": {tokens},\n  \"questions\": {},\n  \"seed\": {seed},\n  \"archive_bytes\": {archive_bytes}\n}}\n",
        bank.len()
    );
    fs::write(meta_path, meta)?;
    println!(
        "generated {} whitespace tokens, {} questions, {} bytes at {}",
        tokens,
        bank.len(),
        archive_bytes,
        out.display()
    );
    Ok(())
}

fn build_index(archive_path: &Path, index_path: &Path, json_path: Option<&Path>) -> io::Result<()> {
    let started = Instant::now();
    let archive = File::open(archive_path)?;
    let mut reader = BufReader::with_capacity(4 * 1024 * 1024, archive);
    let mut map: BTreeMap<String, Location> = BTreeMap::new();
    let mut line = String::new();
    let mut offset = 0_u64;

    loop {
        line.clear();
        let length = reader.read_line(&mut line)?;
        if length == 0 {
            break;
        }
        for key in identifiers(&line) {
            map.insert(
                key,
                Location {
                    offset,
                    length: length as u32,
                },
            );
        }
        offset += length as u64;
    }

    let mut writer = BufWriter::new(File::create(index_path)?);
    writeln!(writer, "{INDEX_HEADER}")?;
    for (key, location) in &map {
        writeln!(writer, "{key}\t{}\t{}", location.offset, location.length)?;
    }
    writer.flush()?;
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let index_bytes = fs::metadata(index_path)?.len();
    let result = format!(
        "{{\n  \"format\": \"pickle-retrieval-index-v1\",\n  \"archive\": \"{}\",\n  \"archive_bytes\": {},\n  \"index\": \"{}\",\n  \"index_bytes\": {},\n  \"identifiers\": {},\n  \"build_seconds\": {:.6},\n  \"archive_mebibytes_per_second\": {:.3}\n}}\n",
        json_escape(&archive_path.display().to_string()),
        offset,
        json_escape(&index_path.display().to_string()),
        index_bytes,
        map.len(),
        elapsed_seconds,
        offset as f64 / (1024.0 * 1024.0) / elapsed_seconds.max(1e-12),
    );
    print!("{result}");
    if let Some(path) = json_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, result)?;
    }
    Ok(())
}

fn load_index(path: &Path) -> io::Result<BTreeMap<String, Location>> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();
    match lines.next().transpose()?.as_deref() {
        Some(INDEX_HEADER) => {}
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported index format",
            ))
        }
    }
    let mut map = BTreeMap::new();
    for line in lines {
        let line = line?;
        let mut fields = line.split('\t');
        let key = fields.next().unwrap_or_default().to_string();
        let offset = fields
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing offset"))?
            .parse::<u64>()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid offset"))?;
        let length = fields
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing length"))?
            .parse::<u32>()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid length"))?;
        if key.is_empty() || !is_identifier(&key) || length == 0 || fields.next().is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid index entry",
            ));
        }
        map.insert(key, Location { offset, length });
    }
    Ok(map)
}

fn read_location(file: &mut File, location: Location) -> io::Result<String> {
    file.seek(SeekFrom::Start(location.offset))?;
    let mut bytes = vec![0_u8; location.length as usize];
    file.read_exact(&mut bytes)?;
    String::from_utf8(bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "archive is not UTF-8"))
}

fn trailing_value<'a>(text: &'a str, marker: &str) -> Option<&'a str> {
    text.rsplit_once(marker)
        .map(|(_, value)| clean_token(value.trim()))
        .filter(|value| !value.is_empty())
}

fn extract_value(text: &str) -> Option<String> {
    trailing_value(text, " is ")
        .or_else(|| trailing_value(text, "stores serial number "))
        .map(str::to_string)
}

fn answer(
    archive: &mut File,
    index: &BTreeMap<String, Location>,
    question: &str,
) -> io::Result<String> {
    let keys = identifiers(question);
    let Some(key) = keys.first() else {
        return Ok(ABSTAIN.into());
    };
    let Some(location) = index.get(key) else {
        return Ok(ABSTAIN.into());
    };
    let first = read_location(archive, *location)?;
    if first.contains("stored under reference") {
        let pointer = identifiers(&first)
            .into_iter()
            .find(|candidate| candidate != key);
        if let Some(pointer) = pointer {
            if let Some(pointer_location) = index.get(&pointer) {
                if pointer_location.offset == location.offset {
                    return Ok(ABSTAIN.into());
                }
                let second = read_location(archive, *pointer_location)?;
                if second.contains("stored under reference") {
                    return Ok(ABSTAIN.into());
                }
                if let Some(value) = extract_value(&second) {
                    return Ok(value);
                }
            }
        }
        return Ok(ABSTAIN.into());
    }
    Ok(extract_value(&first).unwrap_or_else(|| ABSTAIN.into()))
}

fn load_bank(path: &Path) -> io::Result<Vec<Question>> {
    let reader = BufReader::new(File::open(path)?);
    let mut out = Vec::new();
    for (line_number, line) in reader.lines().enumerate() {
        let line = line?;
        if line_number == 0 && line == "task\tquestion\tanswer" {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid bank line {}", line_number + 1),
            ));
        }
        out.push(Question {
            task: fields[0].into(),
            question: fields[1].into(),
            answer: fields[2].into(),
        });
    }
    Ok(out)
}

fn percentile(sorted: &[u128], fraction: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index]
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn benchmark(
    archive_path: &Path,
    index_path: &Path,
    bank_path: &Path,
    iterations: usize,
    concurrency: usize,
    json_path: Option<&Path>,
) -> io::Result<()> {
    if concurrency == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--threads must be positive",
        ));
    }
    let load_started = Instant::now();
    let index = load_index(index_path)?;
    let index_load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let bank = load_bank(bank_path)?;
    if bank.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "question bank is empty",
        ));
    }
    let mut archive = File::open(archive_path)?;
    let mut timings_us = Vec::with_capacity(bank.len() * iterations);
    let mut correct = 0_usize;
    let mut task_results: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut failures = Vec::new();

    for iteration in 0..iterations {
        for item in &bank {
            let started = Instant::now();
            let predicted = answer(&mut archive, &index, &item.question)?;
            timings_us.push(started.elapsed().as_micros());
            if iteration == 0 {
                let entry = task_results.entry(item.task.clone()).or_insert((0, 0));
                entry.1 += 1;
                if predicted == item.answer {
                    correct += 1;
                    entry.0 += 1;
                } else if failures.len() < 20 {
                    failures.push((item.question.clone(), item.answer.clone(), predicted));
                }
            }
        }
    }
    timings_us.sort_unstable();
    let p50 = percentile(&timings_us, 0.50);
    let p95 = percentile(&timings_us, 0.95);
    let mean = timings_us.iter().sum::<u128>() as f64 / timings_us.len().max(1) as f64;
    let accuracy = correct as f64 / bank.len().max(1) as f64;

    let mut reopen_timings_us = Vec::with_capacity(bank.len());
    let mut reopen_correct = 0_usize;
    for item in &bank {
        let started = Instant::now();
        let mut reopened = File::open(archive_path)?;
        let predicted = answer(&mut reopened, &index, &item.question)?;
        reopen_timings_us.push(started.elapsed().as_micros());
        reopen_correct += usize::from(predicted == item.answer);
    }
    reopen_timings_us.sort_unstable();
    let reopen_mean =
        reopen_timings_us.iter().sum::<u128>() as f64 / reopen_timings_us.len() as f64;
    let reopen_p50 = percentile(&reopen_timings_us, 0.50);
    let reopen_p95 = percentile(&reopen_timings_us, 0.95);

    let concurrent_started = Instant::now();
    let total_concurrent_queries = bank.len() * iterations;
    let concurrent_correct = thread::scope(|scope| -> io::Result<usize> {
        let mut handles = Vec::with_capacity(concurrency);
        for worker in 0..concurrency {
            let archive_path = archive_path.to_path_buf();
            let index = &index;
            let bank = &bank;
            handles.push(scope.spawn(move || -> io::Result<usize> {
                let mut archive = File::open(archive_path)?;
                let mut correct = 0;
                for work in (worker..total_concurrent_queries).step_by(concurrency) {
                    let item = &bank[work % bank.len()];
                    if answer(&mut archive, index, &item.question)? == item.answer {
                        correct += 1;
                    }
                }
                Ok(correct)
            }));
        }
        let mut correct = 0;
        for handle in handles {
            correct += handle
                .join()
                .map_err(|_| io::Error::other("retrieval worker panicked"))??;
        }
        Ok(correct)
    })?;
    let concurrent_seconds = concurrent_started.elapsed().as_secs_f64();
    let concurrent_qps = total_concurrent_queries as f64 / concurrent_seconds.max(1e-12);

    let mut tasks_json = String::new();
    for (i, (task, (task_correct, task_total))) in task_results.iter().enumerate() {
        if i > 0 {
            tasks_json.push_str(",\n");
        }
        tasks_json.push_str(&format!(
            "    \"{}\": {{\"correct\": {}, \"total\": {}, \"accuracy\": {:.6}}}",
            json_escape(task),
            task_correct,
            task_total,
            *task_correct as f64 / (*task_total).max(1) as f64
        ));
    }
    let result = format!(
        "{{\n  \"format\": \"pickle-retrieval-result-v2\",\n  \"archive\": \"{}\",\n  \"archive_bytes\": {},\n  \"index_bytes\": {},\n  \"index_entries\": {},\n  \"index_load_ms\": {:.3},\n  \"questions\": {},\n  \"iterations\": {},\n  \"correct\": {},\n  \"accuracy\": {:.6},\n  \"query_latency_us\": {{\"mean\": {:.3}, \"p50\": {}, \"p95\": {}}},\n  \"reopen_latency_us\": {{\"scope\": \"archive reopened for every query; OS page cache not cleared\", \"correct\": {}, \"mean\": {:.3}, \"p50\": {}, \"p95\": {}}},\n  \"concurrent\": {{\"threads\": {}, \"queries\": {}, \"correct\": {}, \"seconds\": {:.6}, \"queries_per_second\": {:.3}}},\n  \"tasks\": {{\n{}\n  }}\n}}\n",
        json_escape(&archive_path.display().to_string()),
        fs::metadata(archive_path)?.len(),
        fs::metadata(index_path)?.len(),
        index.len(),
        index_load_ms,
        bank.len(),
        iterations,
        correct,
        accuracy,
        mean,
        p50,
        p95,
        reopen_correct,
        reopen_mean,
        reopen_p50,
        reopen_p95,
        concurrency,
        total_concurrent_queries,
        concurrent_correct,
        concurrent_seconds,
        concurrent_qps,
        tasks_json
    );

    print!("{result}");
    if let Some(path) = json_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &result)?;
    }
    if !failures.is_empty() {
        eprintln!("failures:");
        for (question, expected, predicted) in failures {
            eprintln!("  {question}\n    expected: {expected}\n    predicted: {predicted}");
        }
    }
    Ok(())
}

fn load_native(opts: &HashMap<String, String>) -> io::Result<native::Model> {
    if let Some(value) = opts.get("threads") {
        let threads = value.parse::<usize>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid --threads: {value}"),
            )
        })?;
        native::configure_threads(threads)?;
    }
    let mut model = native::Model::load(Path::new(required(opts, "model")?))?;
    model.set_kernel(opts.get("kernel").map(String::as_str).unwrap_or("auto"))?;
    Ok(model)
}

fn model_info(opts: &HashMap<String, String>) -> io::Result<()> {
    let model = load_native(opts)?;
    println!(
        "{{\n  \"format\": \"pickle-native-model-v{}\",\n  \"authenticated_sha256\": true,\n  \"authenticated_header_and_body_sha256\": \"{}\",\n  \"native_tokenizer\": \"{}\",\n  \"tied_embeddings\": {},\n  \"default_add_bos\": {},\n  \"kernel\": \"{}\",\n  \"worker_threads\": {},\n  \"model_bytes\": {},\n  \"layers\": {},\n  \"hidden_size\": {},\n  \"intermediate_size\": {},\n  \"vocab_size\": {},\n  \"attention_heads\": {},\n  \"kv_heads\": {},\n  \"head_dim\": {},\n  \"context\": {},\n  \"group_size\": {},\n  \"bos_token_id\": {},\n  \"eos_token_id\": {},\n  \"pad_token_id\": {}\n}}",
        model.format_version,
        model.authenticated_sha256(),
        model.tokenizer_name(),
        model.tied_embeddings,
        model.default_add_bos,
        model.kernel_name(),
        rayon::current_num_threads(),
        model.bytes(),
        model.layers_count,
        model.hidden,
        model.intermediate,
        model.vocab_size,
        model.heads,
        model.kv_heads,
        model.head_dim,
        model.context,
        model.group_size,
        model.bos_token,
        model.eos_token,
        model.pad_token,
    );
    Ok(())
}

fn model_input(
    model: &native::Model,
    opts: &HashMap<String, String>,
    prompt_adds_bos: bool,
) -> io::Result<Vec<u32>> {
    let sources = ["tokens", "prompt", "prompt-file"]
        .into_iter()
        .filter(|key| opts.contains_key(*key))
        .count();
    if sources != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "pass exactly one of --prompt, --prompt-file, or --tokens",
        ));
    }
    if let Some(tokens) = opts.get("tokens") {
        return native::parse_tokens(tokens);
    }
    let prompt = if let Some(prompt) = opts.get("prompt") {
        prompt.clone()
    } else {
        fs::read_to_string(required(opts, "prompt-file")?)?
    };
    let add_bos = parse_u64(
        opts,
        "add-bos",
        Some(if prompt_adds_bos && model.default_add_bos {
            1
        } else {
            0
        }),
    )?;
    if add_bos > 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--add-bos must be 0 or 1",
        ));
    }
    Ok(model.encode(&prompt, add_bos == 1))
}

fn model_tokenize(opts: &HashMap<String, String>) -> io::Result<()> {
    let model = load_native(opts)?;
    let tokens = model_input(&model, opts, false)?;
    let bos_was_added = !opts.contains_key("tokens") && parse_u64(opts, "add-bos", Some(0))? == 1;
    println!(
        "token_ids={}",
        tokens
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    let text_tokens = if bos_was_added { &tokens[1..] } else { &tokens };
    println!("text={}", model.decode(text_tokens));
    Ok(())
}

fn model_logits(opts: &HashMap<String, String>) -> io::Result<()> {
    let model = load_native(opts)?;
    let tokens = model_input(&model, opts, true)?;
    let mut state = model.state(tokens.len())?;
    let logits = native::run_prompt(&model, &mut state, &tokens)?;
    let out = Path::new(required(opts, "out")?);
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(out)?);
    for value in logits {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    println!(
        "wrote {} float32 logits (argmax token {}) to {}",
        logits.len(),
        native::argmax(logits),
        out.display()
    );
    Ok(())
}

fn model_generate(opts: &HashMap<String, String>) -> io::Result<()> {
    let model = load_native(opts)?;
    let tokens = model_input(&model, opts, true)?;
    let steps = parse_u64(opts, "new-tokens", Some(32))? as usize;
    if steps == 0 || tokens.len() + steps > model.context {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "new token count must be positive and fit the context window",
        ));
    }
    let mut state = model.state(tokens.len() + steps)?;
    let mut next = native::argmax(native::run_prompt(&model, &mut state, &tokens)?);
    let mut generated = Vec::with_capacity(steps);
    for step in 0..steps {
        generated.push(next);
        if next == model.eos_token || step + 1 == steps {
            break;
        }
        next = native::argmax(model.forward(&mut state, next, tokens.len() + step)?);
    }
    println!(
        "token_ids={}",
        generated
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("text={}", model.decode(&generated));
    Ok(())
}

struct E2eSample {
    prompt_tokens: usize,
    tokenization_seconds: f64,
    state_allocation_seconds: f64,
    prefill_seconds: f64,
    decode_seconds: f64,
    total_seconds: f64,
    state_bytes: usize,
    generated: Vec<u32>,
}

fn e2e_sample(
    model: &native::Model,
    opts: &HashMap<String, String>,
    steps: usize,
) -> io::Result<E2eSample> {
    let request_started = Instant::now();
    let tokenization_started = Instant::now();
    let tokens = model_input(model, opts, true)?;
    let tokenization_seconds = tokenization_started.elapsed().as_secs_f64();
    if steps == 0 || tokens.len() + steps > model.context {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "new token count must be positive and fit the context window",
        ));
    }

    let allocation_started = Instant::now();
    let mut state = model.state(tokens.len() + steps)?;
    let state_allocation_seconds = allocation_started.elapsed().as_secs_f64();
    let prefill_started = Instant::now();
    let mut next = native::argmax(native::run_prompt(model, &mut state, &tokens)?);
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    let decode_started = Instant::now();
    let mut generated = Vec::with_capacity(steps);
    generated.push(next);
    for step in 1..steps {
        next = native::argmax(model.forward(&mut state, next, tokens.len() + step - 1)?);
        generated.push(next);
    }
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    Ok(E2eSample {
        prompt_tokens: tokens.len(),
        tokenization_seconds,
        state_allocation_seconds,
        prefill_seconds,
        decode_seconds,
        total_seconds: request_started.elapsed().as_secs_f64(),
        state_bytes: state.allocated_bytes(),
        generated,
    })
}

fn model_e2e(opts: &HashMap<String, String>) -> io::Result<()> {
    let model_path = Path::new(required(opts, "model")?);
    let steps = parse_u64(opts, "new-tokens", Some(16))? as usize;
    let iterations = parse_u64(opts, "iterations", Some(3))? as usize;
    if iterations == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "iterations must be positive",
        ));
    }
    let load_started = Instant::now();
    let model = load_native(opts)?;
    let cold_load_seconds = load_started.elapsed().as_secs_f64();
    let cold = e2e_sample(&model, opts, steps)?;
    let mut warm = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        warm.push(e2e_sample(&model, opts, steps)?);
    }
    let average = |value: fn(&E2eSample) -> f64| -> f64 {
        warm.iter().map(value).sum::<f64>() / warm.len() as f64
    };
    let warm_total = average(|sample| sample.total_seconds);
    let warm_ttft = average(|sample| {
        sample.tokenization_seconds + sample.state_allocation_seconds + sample.prefill_seconds
    });
    let result = format!(
        "{{\n  \"format\": \"pickle-native-end-to-end-v1\",\n  \"model\": \"{}\",\n  \"model_bytes\": {},\n  \"kernel\": \"{}\",\n  \"worker_threads\": {},\n  \"prompt_tokens\": {},\n  \"generated_tokens\": {},\n  \"cold\": {{\n    \"model_load_seconds\": {:.6},\n    \"tokenization_seconds\": {:.6},\n    \"state_allocation_seconds\": {:.6},\n    \"prefill_seconds\": {:.6},\n    \"time_to_first_token_seconds\": {:.6},\n    \"decode_after_first_token_seconds\": {:.6},\n    \"total_request_seconds\": {:.6}\n  }},\n  \"warm\": {{\n    \"iterations\": {},\n    \"mean_tokenization_seconds\": {:.6},\n    \"mean_state_allocation_seconds\": {:.6},\n    \"mean_prefill_seconds\": {:.6},\n    \"mean_time_to_first_token_seconds\": {:.6},\n    \"mean_decode_after_first_token_seconds\": {:.6},\n    \"mean_total_request_seconds\": {:.6},\n    \"generated_tokens_per_second\": {:.6}\n  }},\n  \"allocated_model_and_state_bytes\": {},\n  \"generated_token_ids_last_iteration\": [{}]\n}}\n",
        json_escape(&model_path.display().to_string()),
        model.bytes(),
        model.kernel_name(),
        rayon::current_num_threads(),
        cold.prompt_tokens,
        steps,
        cold_load_seconds,
        cold.tokenization_seconds,
        cold.state_allocation_seconds,
        cold.prefill_seconds,
        cold_load_seconds
            + cold.tokenization_seconds
            + cold.state_allocation_seconds
            + cold.prefill_seconds,
        cold.decode_seconds,
        cold_load_seconds + cold.total_seconds,
        iterations,
        average(|sample| sample.tokenization_seconds),
        average(|sample| sample.state_allocation_seconds),
        average(|sample| sample.prefill_seconds),
        warm_ttft,
        average(|sample| sample.decode_seconds),
        warm_total,
        steps as f64 / warm_total,
        model.bytes() + cold.state_bytes,
        warm.last()
            .unwrap()
            .generated
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    print!("{result}");
    if let Some(path) = opts.get("json") {
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &result)?;
    }
    Ok(())
}

fn model_bench(opts: &HashMap<String, String>) -> io::Result<()> {
    let model_path = Path::new(required(opts, "model")?);
    let model = load_native(opts)?;
    let tokens = model_input(&model, opts, true)?;
    let steps = parse_u64(opts, "new-tokens", Some(16))? as usize;
    let iterations = parse_u64(opts, "iterations", Some(1))? as usize;
    if iterations == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "iterations must be positive",
        ));
    }
    let mut total_elapsed = 0.0;
    let mut iteration_rates = Vec::with_capacity(iterations);
    let mut generated = Vec::new();
    let mut state_bytes = 0;
    for _ in 0..iterations {
        let result = native::benchmark_decode(&model, &tokens, steps)?;
        generated = result.0;
        total_elapsed += result.1;
        state_bytes = result.2;
        iteration_rates.push(steps as f64 / result.1);
    }
    let tokens_per_second = (steps * iterations) as f64 / total_elapsed;
    let result = format!(
        "{{\n  \"format\": \"pickle-native-decode-result-v2\",\n  \"model\": \"{}\",\n  \"model_bytes\": {},\n  \"kernel\": \"{}\",\n  \"worker_threads\": {},\n  \"prompt_tokens\": {},\n  \"timed_decode_tokens_per_iteration\": {},\n  \"iterations\": {},\n  \"total_elapsed_seconds\": {:.6},\n  \"tokens_per_second\": {:.6},\n  \"iteration_tokens_per_second\": [{}],\n  \"allocated_model_and_state_bytes\": {},\n  \"generated_token_ids_last_iteration\": [{}]\n}}\n",
        json_escape(&model_path.display().to_string()),
        model.bytes(),
        model.kernel_name(),
        rayon::current_num_threads(),
        tokens.len(),
        steps,
        iterations,
        total_elapsed,
        tokens_per_second,
        iteration_rates
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join(", "),
        model.bytes() + state_bytes,
        generated
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    print!("{result}");
    if let Some(path) = opts.get("json") {
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &result)?;
    }
    Ok(())
}

fn run() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage();
    }
    let command = &args[1];
    let opts = parse_options(&args[2..])?;
    match command.as_str() {
        "generate" => {
            let out = PathBuf::from(required(&opts, "out")?);
            let tokens = parse_u64(&opts, "tokens", None)?;
            let questions = parse_u64(&opts, "questions", Some(200))? as usize;
            let seed = parse_u64(&opts, "seed", Some(20260824))?;
            generate(&out, tokens, questions, seed)
        }
        "index" => build_index(
            Path::new(required(&opts, "archive")?),
            Path::new(required(&opts, "out")?),
            opts.get("json").map(Path::new),
        ),
        "ask" => {
            let index = load_index(Path::new(required(&opts, "index")?))?;
            let mut archive = File::open(required(&opts, "archive")?)?;
            let response = answer(&mut archive, &index, required(&opts, "question")?)?;
            println!("{response}");
            Ok(())
        }
        "bench" => {
            let iterations = parse_u64(&opts, "iterations", Some(5))? as usize;
            let concurrency = parse_u64(&opts, "threads", Some(1))? as usize;
            let json_path = opts.get("json").map(Path::new);
            benchmark(
                Path::new(required(&opts, "archive")?),
                Path::new(required(&opts, "index")?),
                Path::new(required(&opts, "bank")?),
                iterations,
                concurrency,
                json_path,
            )
        }
        "model-info" => model_info(&opts),
        "model-tokenize" => model_tokenize(&opts),
        "model-logits" => model_logits(&opts),
        "model-generate" => model_generate(&opts),
        "model-bench" => model_bench(&opts),
        "model-e2e" => model_e2e(&opts),
        _ => usage(),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifier_detection_is_strict() {
        assert!(is_identifier("Grus-189?"));
        assert!(is_identifier("SN-442976."));
        assert!(is_identifier("(Étoile-442976),"));
        assert!(!is_identifier("ordinary"));
        assert!(!is_identifier("hyphen-only"));
    }

    #[test]
    fn event_bank_has_every_task() {
        let (_, bank) = make_events(1_000_000, 100, 7);
        for task in [
            "direct",
            "lookalike",
            "latest_wins",
            "two_hop",
            "abstain",
            "unicode_identifier",
            "long_value",
            "malformed_record",
            "missing_pointer",
            "punctuation",
            "case_sensitive",
            "adversarial_abstain",
        ] {
            assert!(bank.iter().any(|item| item.task == task));
        }
    }

    #[test]
    fn self_referential_missing_pointer_abstains() {
        let path =
            env::temp_dir().join(format!("pickle50-pointer-test-{}.txt", std::process::id()));
        let line =
            "The record for device Pointer-123456 is stored under reference Missing-654321.\n";
        fs::write(&path, line).unwrap();
        let location = Location {
            offset: 0,
            length: line.len() as u32,
        };
        let index = BTreeMap::from([
            ("Pointer-123456".to_string(), location),
            ("Missing-654321".to_string(), location),
        ]);
        let mut archive = File::open(&path).unwrap();
        let result = answer(
            &mut archive,
            &index,
            "What serial number is stored for device Pointer-123456?",
        )
        .unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(result, ABSTAIN);
    }
}
