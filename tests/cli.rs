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

fn json(args: &[&str], code: i32) -> serde_json::Value {
    let (c, out, err) = ctxmeter(args);
    assert_eq!(c, code, "{args:?} exit code\n{err}");
    serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("{args:?} stdout is not one JSON document: {e}\n{out}"))
}

fn close(v: &serde_json::Value, want: f64, tol: f64) {
    let got = v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"));
    assert!(
        (got - want).abs() <= tol,
        "got {got}, want {want} within {tol}"
    );
}

#[test]
fn summary_json() {
    let v = json(&["summary", "--root", CORPUS, "--json"], 0);
    assert_eq!(v["turns"], 4);
    assert_eq!(v["sessions"], 1);
    assert_eq!(v["tokens"]["fresh"], 24);
    assert_eq!(v["tokens"]["write"], 20250);
    assert_eq!(v["tokens"]["read"], 56500);
    assert_eq!(v["tokens"]["output"], 233);
    close(&v["share"]["read"], 73.59, 0.005);
    close(&v["share"]["write"], 26.38, 0.005);
    close(&v["billed"]["write"], 25312.5, 1e-9);
    close(&v["billed"]["read"], 5650.0, 1e-6);
    close(&v["billed"]["total"], 30986.5, 1e-6);
    close(&v["no_cache_total"], 76774.0, 1e-9);
    close(&v["cache_hit_rate"], 73.59, 0.005);
    close(&v["caching_saves"], 59.64, 0.005);
    assert_eq!(v["models"]["claude-sonnet-5"], 4);
}

#[test]
fn floor_json() {
    let v = json(&["floor", "--root", CORPUS, "--json"], 0);
    assert_eq!(v["n"], 1);
    for q in ["p25", "p50", "p75", "p90"] {
        assert_eq!(v[q], 18012, "{q}");
    }
    assert_eq!(v["months"], serde_json::json!({}));
}
