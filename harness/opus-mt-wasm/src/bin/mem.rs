//! Heap accounting per loading stage: live bytes, and the peak since the previous stage.

use std::alloc::GlobalAlloc;
use std::alloc::Layout;
use std::alloc::System;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

use clap::Parser;
use tract_onnx::prelude::*;
use tract_opus_mt::Error;
use tract_opus_mt::ModelConfig;
use tract_opus_mt::Part;

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        PEAK.fetch_max(LIVE.fetch_add(l.size(), Relaxed) + l.size(), Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        if new > l.size() {
            PEAK.fetch_max(LIVE.fetch_add(new - l.size(), Relaxed) + new - l.size(), Relaxed);
        } else {
            LIVE.fetch_sub(l.size() - new, Relaxed);
        }
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn report(stage: &str) {
    let mb = |b: usize| b as f64 / (1 << 20) as f64;
    println!(
        "{stage:<48} live {:>7.1} MB   peak since last {:>7.1} MB",
        mb(LIVE.load(Relaxed)),
        mb(PEAK.load(Relaxed))
    );
    PEAK.store(LIVE.load(Relaxed), Relaxed);
}

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "models")]
    models: PathBuf,
    #[arg(long)]
    fold: bool,
}

fn main() -> Result<(), Error> {
    let args = Args::parse();
    let cfg = ModelConfig::from_json(&std::fs::read_to_string(args.models.join("config.json"))?)?;
    let mut shared = std::collections::HashMap::new();
    let mut kept = vec![];
    report("start");
    for part in Part::ALL {
        let bytes = std::fs::read(args.models.join(part.file()))?;
        report(&format!("{}: bytes read", part.file()));
        let proto = tract_onnx::onnx().proto_model_for_read(&mut &*bytes)?;
        report("  protobuf decoded");
        drop(bytes);
        report("  file bytes dropped");
        let mut model =
            tract_onnx::onnx().with_ignore_value_info(true).model_for_proto_model(&proto)?;
        report("  InferenceModel built");
        drop(proto);
        report("  protobuf dropped");
        tract_opus_mt::set_input_facts(&mut model, part, &cfg)?;
        let mut typed = model.into_optimized()?;
        report("  optimized (InferenceModel dropped)");
        if args.fold {
            let reused = tract_opus_mt::reuse_folds(&mut typed, &shared)?;
            report(&format!("  reused {reused} folds"));
            shared.extend(tract_opus_mt::fold_all_consts(&mut typed)?);
            report("  folded");
        }
        kept.push(typed.into_runnable()?);
        report("  runnable");
    }
    Ok(())
}
