//! The conformance suite's command: `record` takes Metal's answers on a Mac, `replay` runs our side
//! on a venue and reports.
//!
//! ```text
//! conformance record [case ...]
//! conformance replay [--cpu] [case ...]
//! conformance cases
//! ```
//!
//! With no case named, the whole case set is selected. `--cpu` replays on a CPU Vulkan
//! implementation (lavapipe), which is how the no-Metal leg is proven.
//!
//! The command's exit status is 0 only where the selection is compliant: no divergence, no refusal,
//! no unrecorded case and no stale recording. A venue that lacks a feature a case needs does not
//! spoil it.

use metal2vulkan_conformance::device::Venues;
use metal2vulkan_conformance::run;

fn main() -> std::process::ExitCode {
    match command() {
        Ok(compliant) if compliant => std::process::ExitCode::SUCCESS,
        Ok(_) => std::process::ExitCode::FAILURE,
        Err(why) => {
            eprintln!("conformance: {why}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Run the command, answering whether the selection is compliant.
fn command() -> Result<bool, String> {
    let mut words = std::env::args().skip(1);
    let verb = words.next().unwrap_or_else(|| "replay".to_owned());
    let mut venues = Venues::Gpu;
    let mut names = Vec::new();
    for word in words {
        match word.as_str() {
            "--cpu" => venues = Venues::Cpu,
            "--gpu" => venues = Venues::Gpu,
            flag if flag.starts_with('-') => return Err(format!("unknown flag {flag}")),
            name => names.push(name.to_owned()),
        }
    }
    let cases = run::selected(&names)?;
    match verb.as_str() {
        "cases" => {
            for case in &cases {
                println!("{}  {:?}", case.name(), case.metal_source());
            }
            Ok(true)
        }
        "record" => record(&cases),
        "replay" => {
            let report = run::replay_on(venues, &cases)?;
            print!("{}", report.render());
            Ok(report.is_compliant())
        }
        other => Err(format!(
            "unknown command {other}: expected record, replay or cases"
        )),
    }
}

/// Take the selection's Metal answers, printing where each recording went.
#[cfg(target_os = "macos")]
fn record(cases: &[Box<dyn metal2vulkan_conformance::case::Case>]) -> Result<bool, String> {
    let mut all = true;
    for (case, written) in run::record_all(cases) {
        match written {
            Ok(path) => println!("{case}: {}", path.display()),
            Err(why) => {
                all = false;
                println!("{case}: {why}");
            }
        }
    }
    Ok(all)
}

/// There is no Metal to record from off macOS.
#[cfg(not(target_os = "macos"))]
fn record(_cases: &[Box<dyn metal2vulkan_conformance::case::Case>]) -> Result<bool, String> {
    Err("record mode runs on macOS, where Metal is".to_owned())
}
