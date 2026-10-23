//! The conformance suite's command: `record` takes Metal's answers on a Mac, `replay` runs our side
//! on a venue and reports.
//!
//! ```text
//! conformance record [case ...]
//! conformance replay [--cpu] [--default-policy] [case ...]
//! conformance cases
//! ```
//!
//! `record` runs each case on Metal twice: plain, which is the behavioural oracle, and once under
//! Metal's own API validation layer, in a child of this command ([`metal2vulkan_conformance::validated`]).
//!
//! With no case named, the whole case set is selected. `--cpu` replays on a CPU Vulkan
//! implementation (lavapipe), which is how the no-Metal leg is proven. A replay holds the runtime
//! to exact answers by default; `--default-policy` replays under the runtime's own defaults, the
//! fast form of every speed-for-accuracy choice, which is what a consumer runs. Comparing a case
//! under both is how the suite says whether a difference is the policy's or the implementation's.
//!
//! The command's exit status is 0 only where the selection holds no surprise: no divergence, no
//! unrecorded case, no stale recording, and exactly the refusals `known-refusals.tsv` expects — a
//! refusal it does not list, and a row it lists that is no longer refused, each fail the run. A
//! venue that lacks a feature a case needs does not spoil it.
//!
//! The ledger never changes a verdict. A refusal it expects is still printed as `Refused` and
//! still counted, because its rows are open bugs rather than accepted differences.

use metal2vulkan_conformance::device::{Policy, Venues};
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
    let mut policy = Policy::Exact;
    let mut names = Vec::new();
    for word in words {
        match word.as_str() {
            "--cpu" => venues = Venues::Cpu,
            "--gpu" => venues = Venues::Gpu,
            "--default-policy" => policy = Policy::Default,
            "--exact-policy" => policy = Policy::Exact,
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
        metal2vulkan_conformance::validated::LEG_VERB => validated_leg(&cases),
        "replay" => {
            let report = run::replay_on_under(venues, policy, &cases)?;
            print!("{}", report.render());
            // The known-refusals ledger decides the exit status and nothing else: each of its rows
            // is still printed above as a `Refused` row and still counted there. It is held in
            // both directions, so a new refusal fails the run and a row that is no longer refused
            // fails it too.
            let ledger = metal2vulkan_conformance::known_refusals::ledger()?;
            let unexpected = report.unexpected(&ledger);
            if unexpected.is_empty() {
                let expected = ledger
                    .iter()
                    .filter(|known| report.rows().iter().any(|row| row.case == known.case))
                    .count();
                if expected > 0 {
                    println!(
                        "\nthe {expected} refusal(s) above are exactly the open gaps \
                         known-refusals.tsv expects, each one a bug with a todo behind it"
                    );
                }
                return Ok(true);
            }
            println!();
            for why in &unexpected {
                println!("not what known-refusals.tsv expects: {why}");
            }
            Ok(false)
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

/// Take one case's validated leg: the body of the child process record mode spawns with Metal's
/// API validation layer on. It is not part of the command's documented surface.
#[cfg(target_os = "macos")]
fn validated_leg(cases: &[Box<dyn metal2vulkan_conformance::case::Case>]) -> Result<bool, String> {
    let [case] = cases else {
        return Err(format!(
            "{}: one case at a time",
            metal2vulkan_conformance::validated::LEG_VERB
        ));
    };
    run::validated_leg(case.as_ref())?;
    Ok(true)
}

/// There is no Metal to validate against off macOS.
#[cfg(not(target_os = "macos"))]
fn validated_leg(_cases: &[Box<dyn metal2vulkan_conformance::case::Case>]) -> Result<bool, String> {
    Err("Metal's API validation layer is on macOS, where Metal is".to_owned())
}
