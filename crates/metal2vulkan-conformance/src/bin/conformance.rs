//! The conformance suite's command: `record` takes Metal's answers on a Mac, `replay` runs our side
//! on a venue and reports.
//!
//! ```text
//! conformance record [case ...]
//! conformance replay [--cpu] [--default-policy] [case ...]
//! conformance cost [--cpu] [--default-policy] [--runs N] [--write] [--show] [--alone] [case ...]
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
//!
//! `cost` is the other half ([`metal2vulkan_conformance::cost`]): it runs our side alone and says
//! what the runtime did to answer each case, by count. `--runs N` runs the whole selection N times
//! (eight by default), each round on a fresh device, so a count that is not deterministic is seen
//! to move rather than assumed; `--write` puts the bounds in the cases' own recordings, and
//! `--show` prints every count a case moved, which is what a contrast run reads. `replay` then
//! holds each case
//! to them and reports the cost verdicts in their own section. Counts are ours and not Metal's, so
//! they never touch a correctness verdict — but a case going over its bounds does fail the run, on
//! its own line of the exit status.

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
    // The rounds `cost` takes when nothing says otherwise. Eight, measured: a count the
    // measurement never sees move is gated, and `host_texel_copies` in
    // `chroma-pair-texel-reads-bgrg422` held at 2 over three rounds and over two whole-set
    // replays, then reached 3 — and running that case alone for eight rounds moved it 2..3. A rare
    // second arm is the one thing a thin measurement turns into a false regression.
    let mut runs = 8;
    let mut write = false;
    let mut show = false;
    let mut alone = false;
    let mut words = words.peekable();
    while let Some(word) = words.next() {
        match word.as_str() {
            "--cpu" => venues = Venues::Cpu,
            "--gpu" => venues = Venues::Gpu,
            "--default-policy" => policy = Policy::Default,
            "--exact-policy" => policy = Policy::Exact,
            "--write" => write = true,
            "--show" => show = true,
            "--alone" => alone = true,
            "--runs" => {
                let value = words.next().ok_or("--runs wants a number")?;
                runs = value
                    .parse()
                    .map_err(|_| format!("--runs {value}: not a number"))?;
            }
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
        "cost" => cost(venues, policy, &cases, runs, write, show, alone),
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
            // The cost half is a gate of its own, so it has its own line in the exit status: a
            // count going over its bound is never a conformance difference, and a conformance
            // difference is never a cost result. Both must hold for the run to be green.
            let within = report.cost_is_within();
            if !within {
                println!(
                    "\n{} case(s) did more work than the measurement their bounds name; the \
                     counts are listed above",
                    report.cost_overruns().len()
                );
            }
            if unexpected.is_empty() && within {
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
            "unknown command {other}: expected record, replay, cost or cases"
        )),
    }
}

/// Measure what the selection costs on a venue and print it, writing the bounds into the cases'
/// own recordings with `--write`.
///
/// The whole selection runs `--runs` times in order, because a case's counts depend on what ran
/// before it in the same process: the runtime pools command pools, memory blocks and descriptor
/// sets across cases, so the measurement must have the shape the gate will have.
///
/// `--alone` measures the other shape instead ([`run::cost_alone_on_under`]): each case on a
/// device nothing else has touched. That is where a pooled count is at its largest, so it is the
/// bound a later landing cannot invalidate — and the contrast between the two shapes is what says
/// whether a count is the case's own work or the selection's.
fn cost(
    venues: Venues,
    policy: Policy,
    cases: &[Box<dyn metal2vulkan_conformance::case::Case>],
    runs: u32,
    write: bool,
    show: bool,
    alone: bool,
) -> Result<bool, String> {
    // A bound is written from both shapes or not at all: neither bounds the other, and gating a
    // count the two disagree on is what makes a bound go stale on the next landing.
    let (venue, measured) = if write {
        run::cost_contrast_on_under(venues, policy, cases, runs)?
    } else if alone {
        run::cost_alone_on_under(venues, policy, cases, runs)?
    } else {
        run::cost_on_under(venues, policy, cases, runs)?
    };
    let words = if write {
        format!(
            "{runs} run(s) of the whole selection on {venue}, in the case set's own order, and \
             {runs} more of each case alone; the largest each count reached, gated only where the \
             two shapes agreed"
        )
    } else if alone {
        format!(
            "{runs} run(s) of each case on its own on {venue}, nothing before it, the largest \
             each count reached"
        )
    } else {
        format!(
            "{runs} run(s) of the whole selection on {venue}, in the case set's own order, the \
             largest each count reached"
        )
    };
    println!(
        "cost on {venue}, over {runs} run(s) of {} case(s)",
        cases.len()
    );
    println!("{words}\n");
    let mut moved = 0;
    let mut raced = 0;
    let mut composed = 0;
    let mut stopped = 0;
    for one in &measured {
        let regions = metal2vulkan_conformance::cost::by_region(&one.counts);
        let region = |region: metal2vulkan_conformance::cost::Region| {
            regions.get(&region).copied().unwrap_or(0)
        };
        use metal2vulkan_conformance::cost::Region;
        println!(
            "{:<44} {:>4} counts: transl {:>3}  api {:>4}  submit {:>4}   {:>8.3} ms",
            one.case,
            one.counts.len(),
            region(Region::Translation),
            region(Region::DeviceApi),
            region(Region::Submission),
            one.nanos as f64 / 1e6,
        );
        if let Some(why) = &one.stopped {
            stopped += 1;
            println!("    the run stopped: {why}");
        }
        let apart = one.composed();
        if !apart.is_empty() {
            composed += apart.len();
            println!(
                "    {} count(s) read differently with the case alone, so the selection decides \
                 them and they are reported and not gated:",
                apart.len()
            );
            for (count, inside, alone) in apart {
                println!("      {count} {inside} in the selection, {alone} alone");
            }
        }
        let spread = one.moved();
        if !spread.is_empty() {
            moved += spread.len();
            println!(
                "    {} count(s) moved between runs for no reason the race rule names, so they \
                 are reported and not gated:",
                spread.len()
            );
            for (count, least, most) in spread {
                println!("      {count} {least}..{most}");
            }
        }
        raced += one.moved_by_a_race().len();
        // `--show` prints every count the case moved. What a contrast run needs: the same case in
        // a selection of one and inside the whole set, count by count.
        if show {
            for (count, value) in one.counts.iter() {
                println!("    = {count} {value}");
            }
        }
    }
    // How much of the runtime's counted surface the selection reaches: the cost half's own
    // coverage number, tracked the way "unrecorded" is rather than left to a feeling.
    let kept = metal2vulkan_conformance::cost::Counts::of(&Default::default())
        .map(|all| all.len())
        .unwrap_or_default();
    let mut touched: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for one in &measured {
        touched.extend(one.counts.iter().map(|(name, _)| name));
    }
    println!(
        "\n{} case(s) measured, {stopped} stopped before finishing, {moved} count reading(s) \
         moved between runs for a reason the race rule does not name, {raced} for one it does, \
         {composed} read differently with the case alone",
        measured.len()
    );
    println!(
        "the selection moves {} of the {kept} counts the runtime keeps",
        touched.len()
    );
    if !write {
        println!("nothing written; pass --write to put these bounds in the recordings");
        return Ok(true);
    }
    let mut all = true;
    for (case, written) in run::record_costs(&venue, &measured, &words) {
        match written {
            Ok(path) => println!("{case}: {}", path.display()),
            Err(why) => {
                all = false;
                println!("{case}: not recorded: {why}");
            }
        }
    }
    Ok(all)
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
