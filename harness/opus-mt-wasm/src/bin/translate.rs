//! Native run of the harness: load time, ms per generated token and peak RSS.

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use tract_opus_mt::Error;
use tract_opus_mt::Loader;
use tract_opus_mt::Part;
use tract_opus_mt::SENTENCES;

#[derive(Parser)]
struct Args {
    /// Directory holding the three model parts, `config.json` and `tokenizer.json`.
    #[arg(long, default_value = "models")]
    models: PathBuf,
    /// Fold constant weights the optimizer leaves to be recomputed every step.
    #[arg(long)]
    fold: bool,
    /// Load the merged decoder rather than the split pair.
    #[arg(long)]
    merged: bool,
    #[arg(long, default_value_t = 3)]
    reps: usize,
    /// Text to translate; defaults to the benchmark sentences.
    #[arg(long = "text")]
    texts: Vec<String>,
}

fn main() -> Result<(), Error> {
    let args = Args::parse();
    let read = |f: &str| std::fs::read_to_string(args.models.join(f));
    let t = Instant::now();
    let mut loader = Loader::new(&read("config.json")?, args.fold)?;
    for &part in Part::set(args.merged) {
        loader.add(part, &std::fs::read(args.models.join(part.file()))?)?;
    }
    let translator = loader.finish(&read("tokenizer.json")?)?;
    println!("load {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let texts: Vec<String> = if args.texts.is_empty() {
        SENTENCES.iter().map(|s| s.to_string()).collect()
    } else {
        args.texts
    };
    for _ in 0..args.reps {
        for text in &texts {
            let t = Instant::now();
            let r = translator.translate(text)?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            println!(
                "{} tok {ms:.0} ms {:.1} ms/tok => {}",
                r.tokens,
                ms / r.tokens as f64,
                r.text
            );
        }
    }
    if let Some(line) =
        std::fs::read_to_string("/proc/self/status")?.lines().find(|l| l.starts_with("VmHWM"))
    {
        println!("{line}");
    }
    Ok(())
}
