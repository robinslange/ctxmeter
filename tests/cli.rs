use std::process::Command;

const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/corpus");

fn ctxmeter(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_ctxmeter"))
        .args(args)
        .env_remove("ANTHROPIC_API_KEY")
        .output()
        .expect("ctxmeter runs");
    (
        out.status.code().expect("exited rather than signalled"),
        String::from_utf8(out.stdout).expect("stdout is utf-8"),
        String::from_utf8(out.stderr).expect("stderr is utf-8"),
    )
}

fn golden(name: &str) -> String {
    let path = format!("{}/tests/golden/{name}.txt", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn text(args: &[&str], name: &str, code: i32) {
    let (c, out, err) = ctxmeter(args);
    assert_eq!(c, code, "{args:?} exit code\n{err}");
    assert_eq!(out, golden(name), "{args:?} text output moved");
}

#[test]
fn summary_text_is_unchanged() {
    text(&["summary", "--root", CORPUS], "summary", 0);
}

#[test]
fn floor_text_is_unchanged() {
    text(&["floor", "--root", CORPUS], "floor", 0);
}

#[test]
fn probes_text_is_unchanged() {
    text(&["probes", "--root", CORPUS], "probes", 0);
}

#[test]
fn an_empty_harvest_text_is_unchanged() {
    text(
        &["probes", "--root", CORPUS, "--max-df", "0"],
        "probes-empty",
        1,
    );
}

#[test]
fn tradeoff_text_is_unchanged() {
    text(&["tradeoff", "--root", CORPUS], "tradeoff", 0);
}

#[test]
fn robustness_text_is_unchanged() {
    text(&["robustness", "--root", CORPUS], "robustness", 0);
}

#[test]
fn sensitivity_text_is_unchanged() {
    text(&["sensitivity", "--root", CORPUS], "sensitivity", 0);
}

#[test]
fn counterfactual_with_nothing_usable_text_is_unchanged() {
    text(
        &[
            "counterfactual",
            "--root",
            CORPUS,
            "--dry-run",
            "--sample",
            "2",
        ],
        "counterfactual-none",
        1,
    );
}

#[test]
fn counterfactual_dry_run_text_is_unchanged() {
    text(
        &[
            "counterfactual",
            "--root",
            CORPUS,
            "--dry-run",
            "--sample",
            "2",
            "--keep-last",
            "1",
        ],
        "counterfactual-dry-run",
        0,
    );
}
