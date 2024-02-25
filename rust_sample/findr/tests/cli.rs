use assert_cmd::Command;
use predicates::prelude::*;
use rand::{distributions::Alphanumeric, Rng};
use std::{borrow::Cow, fs, path::Path};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const RPG: &str = "findr";

fn gen_bad_file() -> String {
    loop {
        let filename: String = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(7)
            .map(char::from)
            .collect();

        if fs::metadata(&filename).is_err() {
            return filename;
        }
    }
}

#[test]
fn skips_bad_dir() -> TestResult {
    let bad = gen_bad_file();
    let expected = format!("{}: .* [(]os error [23][)]", &bad);
    Command::cargo_bin(RPG)?
        .arg(&bad)
        .assert()
        .success()
        .stderr(predicates::str::is_match(expected)?);
    Ok(())
}

fn format_file_name(expected_file: &str) -> Cow<str> {
    expected_file.into()
}

fn run(args: &[&str], expected_file: &str) -> TestResult {
    let file = format_file_name(expected_file);
    let contents = fs::read_to_string(file.as_ref())?;
    let mut expected: Vec<&str> = contents.split("\n").filter(|s| !s.is_empty()).collect();
    expected.sort();

    let cmd = Command::cargo_bin(RPG)?.args(args).assert().success();
    let out = cmd.get_output();
    let stdout = String::from_utf8(out.stdout.clone())?;
    let mut lines: Vec<&str> = stdout.split("\n").filter(|s| !s.is_empty()).collect();
    lines.sort();

    assert_eq!(lines, expected);

    Ok(())
}

#[test]
fn path1() -> TestResult {
    run(&["tests/inputs"], "tests/expected/path1.txt")
}
