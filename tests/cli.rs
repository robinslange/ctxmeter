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

const POLICIES: [&str; 10] = [
    "keep_last_1",
    "keep_last_3",
    "keep_last_5",
    "keep_last_10",
    "keep_last_25",
    "keep_last_50",
    "tail_budget_10k",
    "tail_budget_40k",
    "tail_budget_100k",
    "tail_budget_200k",
];

fn keys(v: &serde_json::Value) -> Vec<&str> {
    v.as_object()
        .unwrap_or_else(|| panic!("not an object: {v}"))
        .keys()
        .map(|k| k.as_str())
        .collect()
}

#[test]
fn probes_json() {
    let v = json(&["probes", "--root", CORPUS, "--json"], 0);
    assert_eq!(v["sessions_scanned"], 1);
    assert_eq!(v["sessions_yielding"], 1);
    assert_eq!(v["probes"], 1);
    assert_eq!(v["max_df"], 3);
    assert_eq!(v["min_gap"], 5);
    close(&v["per_session"], 1.0, 1e-9);
    assert_eq!(v["gap"]["p50"], 8);
    assert_eq!(v["depth"]["p50"], 7);
    assert_eq!(keys(&v["policies"]), POLICIES);
    let p = &v["policies"];
    assert_eq!(p["keep_last_1"]["probes"], 1);
    assert_eq!(p["keep_last_1"]["destroyed"], 1);
    close(&p["keep_last_1"]["retained"], 0.0, 1e-9);
    close(&p["keep_last_3"]["retained"], 100.0, 1e-9);
    close(&p["keep_last_3"]["shallow"], 100.0, 1e-9);
    assert!(p["keep_last_3"]["deep"].is_null());
    assert_eq!(p["keep_last_3"]["kind"], "keep_last");
    assert_eq!(p["keep_last_3"]["keep_last"], 3);
    assert_eq!(p["tail_budget_10k"]["kind"], "tail_budget");
    assert_eq!(p["tail_budget_10k"]["budget_tokens"], 10000);
}

#[test]
fn an_empty_harvest_leaves_stdout_empty_in_json_mode() {
    let (c, out, err) = ctxmeter(&["probes", "--root", CORPUS, "--max-df", "0", "--json"]);
    assert_eq!(c, 1);
    assert_eq!(out, "");
    assert!(err.contains("NO PROBES HARVESTED"), "{err}");
}

#[test]
fn tradeoff_json() {
    let v = json(&["tradeoff", "--root", CORPUS, "--json"], 0);
    assert_eq!(v["sessions"], 1);
    assert_eq!(v["probes"], 1);
    close(&v["baseline_billed"], 28610.0, 0.5);
    assert_eq!(keys(&v["policies"]), POLICIES);
    let k1 = &v["policies"]["keep_last_1"];
    close(&k1["cost_saved"], -0.2, 0.05);
    close(&k1["retained"], 0.0, 1e-9);
    close(&k1["lost"], 100.0, 1e-9);
    assert_eq!(k1["kind"], "keep_last");
}

#[test]
fn robustness_json() {
    let v = json(&["robustness", "--root", CORPUS, "--json"], 0);
    for table in ["cost_model", "estimator", "uncertainty"] {
        assert_eq!(keys(&v[table]), POLICIES, "{table}");
    }
    close(
        &v["cost_model"]["keep_last_1"]["longest_prefix"],
        -0.2,
        0.05,
    );
    close(
        &v["cost_model"]["keep_last_1"]["all_or_nothing"],
        -0.5,
        0.05,
    );
    close(&v["estimator"]["keep_last_1"]["x1"], -0.2, 0.05);
    close(&v["estimator"]["keep_last_1"]["x2"], -0.1, 0.05);
    let u = &v["uncertainty"]["keep_last_3"];
    close(&u["retained"], 100.0, 1e-9);
    close(&u["ci_lo"], 100.0, 1e-9);
    close(&u["ci_hi"], 100.0, 1e-9);
    assert_eq!(u["sessions"], 1);
}

#[test]
fn sensitivity_json() {
    let v = json(&["sensitivity", "--root", CORPUS, "--json"], 0);
    let conditions = v["conditions"].as_array().expect("an array");
    assert_eq!(conditions.len(), 3);
    assert_eq!(conditions[0]["max_df"], 1);
    assert_eq!(conditions[0]["min_gap"], 5);
    assert_eq!(conditions[0]["probes"], 1);
    assert_eq!(keys(&conditions[0]["retained"]), POLICIES);
    close(&conditions[0]["retained"]["keep_last_1"], 0.0, 1e-9);
    assert_eq!(v["ranking"][0], "keep_last_3");
    assert_eq!(v["ranking"][9], "keep_last_1");
    assert_eq!(v["moved"], 0);
    assert_eq!(v["stable"], true);
}

#[test]
fn retention_and_cost_agree_across_commands() {
    let probes = json(&["probes", "--root", CORPUS, "--json"], 0);
    let tradeoff = json(&["tradeoff", "--root", CORPUS, "--json"], 0);
    let robustness = json(&["robustness", "--root", CORPUS, "--json"], 0);
    assert_eq!(probes["probes"], tradeoff["probes"]);
    assert_eq!(probes["sessions_scanned"], tradeoff["sessions"]);
    for p in POLICIES {
        let r = probes["policies"][p]["retained"].as_f64().unwrap();
        close(&tradeoff["policies"][p]["retained"], r, 1e-9);
        close(&robustness["uncertainty"][p]["retained"], r, 1e-9);
        let saved = tradeoff["policies"][p]["cost_saved"].as_f64().unwrap();
        close(&robustness["cost_model"][p]["longest_prefix"], saved, 1e-9);
    }
}

#[test]
fn counterfactual_json_reports_selection_when_nothing_is_usable() {
    let (c, out, err) = ctxmeter(&[
        "counterfactual",
        "--root",
        CORPUS,
        "--dry-run",
        "--sample",
        "2",
        "--json",
    ]);
    assert_eq!(c, 1);
    assert!(err.contains("No usable turns."), "{err}");
    let v: serde_json::Value = serde_json::from_str(&out).expect("one document");
    assert_eq!(v["mode"], "none");
    assert_eq!(v["policy"], "keep_last_3");
    assert_eq!(v["model"], "sonnet");
    assert_eq!(v["skip"], 0);
    assert_eq!(v["selection"]["other_model"], 0);
    assert_eq!(v["selection"]["turns_considered"], 1);
    assert_eq!(v["selection"]["facts_considered"], 1);
    assert_eq!(v["selection"]["policy_kept_the_fact"], 1);
    assert_eq!(v["selection"]["turns_built"], 0);
}

#[test]
fn counterfactual_dry_run_json() {
    let v = json(
        &[
            "counterfactual",
            "--root",
            CORPUS,
            "--dry-run",
            "--sample",
            "2",
            "--keep-last",
            "1",
            "--json",
        ],
        0,
    );
    assert_eq!(v["mode"], "dry_run");
    assert_eq!(v["policy"], "keep_last_1");
    assert_eq!(v["selection"]["turns_built"], 1);
    assert_eq!(v["estimate"]["input_tokens"], 859);
    close(&v["estimate"]["ceiling_usd"], 0.17, 0.005);
    assert_eq!(v["dry_run"]["invalid_arms"], 0);
    assert_eq!(v["dry_run"]["turns"], 1);
    assert_eq!(v["dry_run"]["sessions"], 1);
    assert_eq!(v["dry_run"]["facts"], 1);
}

#[test]
fn counterfactual_dry_run_narrative_moves_to_stderr_in_json_mode() {
    let (_, _, err) = ctxmeter(&[
        "counterfactual",
        "--root",
        CORPUS,
        "--dry-run",
        "--sample",
        "2",
        "--keep-last",
        "1",
        "--json",
    ]);
    assert_eq!(err, golden("counterfactual-dry-run"));
}

#[test]
fn a_refusal_prints_no_document() {
    let (c, out, err) = ctxmeter(&[
        "counterfactual",
        "--root",
        CORPUS,
        "--sample",
        "2",
        "--keep-last",
        "1",
        "--json",
    ]);
    assert_eq!(c, 2);
    assert_eq!(out, "");
    assert!(err.contains("ANTHROPIC_API_KEY is not set"), "{err}");
}
