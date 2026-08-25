//! Size-focused inference-only CLI. Retrieval generation and indexing live in the full binary.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

mod native;
mod sha256;
mod tokenizer;

fn usage() -> ! {
    eprintln!(
        "Usage:\n\
         pickle50-infer model-info --model FILE [--threads N] [--kernel auto|scalar|avx2]\n\
         pickle50-infer model-generate --model FILE (--prompt TEXT | --prompt-file FILE | --tokens IDS) [--new-tokens N] [--threads N] [--kernel auto|scalar|avx2]\n\
         pickle50-infer model-bench --model FILE (--prompt TEXT | --prompt-file FILE | --tokens IDS) [--new-tokens N] [--iterations N] [--threads N] [--kernel auto|scalar|avx2]"
    );
    std::process::exit(2)
}

fn options(args: &[String]) -> io::Result<HashMap<String, String>> {
    if args.len() % 2 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "options require values",
        ));
    }
    let mut output = HashMap::new();
    for pair in args.chunks_exact(2) {
        let key = pair[0]
            .strip_prefix("--")
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "expected --name value"))?;
        output.insert(key.to_string(), pair[1].clone());
    }
    Ok(output)
}

fn required<'a>(opts: &'a HashMap<String, String>, key: &str) -> io::Result<&'a str> {
    opts.get(key)
        .map(String::as_str)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("missing --{key}")))
}

fn number(opts: &HashMap<String, String>, key: &str, default: usize) -> io::Result<usize> {
    opts.get(key).map_or(Ok(default), |value| {
        value.parse::<usize>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid --{key}: {value}"),
            )
        })
    })
}

fn load(opts: &HashMap<String, String>) -> io::Result<native::Model> {
    native::configure_threads(number(opts, "threads", 1)?)?;
    let mut model = native::Model::load(Path::new(required(opts, "model")?))?;
    model.set_kernel(opts.get("kernel").map(String::as_str).unwrap_or("auto"))?;
    Ok(model)
}

fn input(model: &native::Model, opts: &HashMap<String, String>) -> io::Result<Vec<u32>> {
    let sources = ["tokens", "prompt", "prompt-file"]
        .iter()
        .filter(|key| opts.contains_key(**key))
        .count();
    if sources != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "pass exactly one of --prompt, --prompt-file, or --tokens",
        ));
    }
    if let Some(ids) = opts.get("tokens") {
        return native::parse_tokens(ids);
    }
    let text = match opts.get("prompt") {
        Some(value) => value.clone(),
        None => fs::read_to_string(required(opts, "prompt-file")?)?,
    };
    let add_bos = number(opts, "add-bos", usize::from(model.default_add_bos))?;
    if add_bos > 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--add-bos must be 0 or 1",
        ));
    }
    Ok(model.encode(&text, add_bos == 1))
}

fn generated_ids(values: &[u32]) -> String {
    values
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn model_info(opts: &HashMap<String, String>) -> io::Result<()> {
    let model = load(opts)?;
    println!(
        "{{\"format\":\"pickle-native-model-v{}\",\"authenticated_sha256\":true,\"authenticated_header_and_body_sha256\":\"{}\",\"native_tokenizer\":\"{}\",\"tied_embeddings\":{},\"default_add_bos\":{},\"model_bytes\":{},\"kernel\":\"{}\",\"worker_threads\":{},\"context\":{},\"vocab_size\":{}}}",
        model.format_version,
        model.authenticated_sha256(),
        model.tokenizer_name(),
        model.tied_embeddings,
        model.default_add_bos,
        model.bytes(),
        model.kernel_name(),
        rayon::current_num_threads(),
        model.context,
        model.vocab_size,
    );
    Ok(())
}

fn model_generate(opts: &HashMap<String, String>) -> io::Result<()> {
    let model = load(opts)?;
    let tokens = input(&model, opts)?;
    let steps = number(opts, "new-tokens", 32)?;
    if steps == 0 || tokens.len() + steps > model.context {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "generation exceeds context",
        ));
    }
    let mut state = model.state(tokens.len() + steps)?;
    let mut next = native::argmax(native::run_prompt(&model, &mut state, &tokens)?);
    let mut generated = Vec::with_capacity(steps);
    for step in 0..steps {
        generated.push(next);
        if step + 1 < steps {
            next = native::argmax(model.forward(&mut state, next, tokens.len() + step)?);
        }
    }
    println!("token_ids={}", generated_ids(&generated));
    println!("text={}", model.decode(&generated));
    Ok(())
}

fn model_bench(opts: &HashMap<String, String>) -> io::Result<()> {
    let load_started = Instant::now();
    let model = load(opts)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let tokenize_started = Instant::now();
    let tokens = input(&model, opts)?;
    let tokenize_seconds = tokenize_started.elapsed().as_secs_f64();
    let steps = number(opts, "new-tokens", 16)?;
    let iterations = number(opts, "iterations", 3)?;
    if iterations == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "iterations must be positive",
        ));
    }
    let mut decode_seconds = 0.0;
    let mut generated = Vec::new();
    let prefill_started = Instant::now();
    let mut first_state = model.state(tokens.len() + steps)?;
    native::run_prompt(&model, &mut first_state, &tokens)?;
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    for _ in 0..iterations {
        let result = native::benchmark_decode(&model, &tokens, steps)?;
        generated = result.0;
        decode_seconds += result.1;
    }
    let decode_rate = (steps * iterations) as f64 / decode_seconds;
    println!(
        "{{\"format\":\"pickle-inference-benchmark-v1\",\"model_bytes\":{},\"runtime_bytes\":{},\"kernel\":\"{}\",\"worker_threads\":{},\"prompt_tokens\":{},\"generated_tokens_per_iteration\":{},\"iterations\":{},\"cold_load_seconds\":{:.6},\"tokenization_seconds\":{:.6},\"prefill_seconds\":{:.6},\"time_to_first_token_seconds\":{:.6},\"decode_seconds\":{:.6},\"decode_tokens_per_second\":{:.6},\"generated_token_ids\":[{}]}}",
        model.bytes(),
        env::current_exe()?.metadata()?.len(),
        model.kernel_name(),
        rayon::current_num_threads(),
        tokens.len(),
        steps,
        iterations,
        load_seconds,
        tokenize_seconds,
        prefill_seconds,
        load_seconds + tokenize_seconds + prefill_seconds,
        decode_seconds,
        decode_rate,
        generated_ids(&generated),
    );
    Ok(())
}

fn run() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage();
    }
    let opts = options(&args[2..])?;
    match args[1].as_str() {
        "model-info" => model_info(&opts),
        "model-generate" => model_generate(&opts),
        "model-bench" => model_bench(&opts),
        _ => usage(),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
