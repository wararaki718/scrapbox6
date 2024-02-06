use assert_cmd::Command;
use predicates::prelude::*;
use rand::{distributions::Alphanumeric, Rng};
use std::fs;
use tempfile::NamedTempFile;

type TestResult = Result<(), Box<dyn std::error::Error>>;

struct Test {
    input: &'static str,
    out: &'static str,
    out_count: &'static str,
}

const PRG: &str = "uniqr";

const EMPTY: Test = Test {
    input: "tests/inputs/empty.txt",
    out: "tests/inputs/empty.txt.out",
    out_count: "tests/inputs/empty.txt.c.out",
};


fn run(test: &Test) -> TestResult {
    let expected = fs::read_to_string(test.out)?;
    Command::cargo_bin(PRG)?
        .arg(test.input)
        .assert()
        .success()
        .stdout(expected);
    Ok(())
}


fn run_count(test: &Test) -> TestResult {
    let expected = fs::read_to_string(test.out_count)?;
    Command::cargo_bin(PRG)?
        .args(&[test.input, "-c"])
        .assert()
        .success()
        .stdout(expected);
    Ok(())
}


fn run_stdio(test: &Test) -> TestResult {
    let input = fs::read_to_string(test.input)?;
    let expected = fs::read_to_string(test.out)?;
    Command::cargo_bin(PRG)?
        .write_stdin(input)
        .assert()
        .success()
        .stdout(expected);
    Ok(())
}


fn run_stdio_count(test: &Test) -> TestResult {
    let input = fs::read_to_string(test.input)?;
    let expected = fs::read_to_string(test.out_count)?;
    Command::cargo_bin(PRG)?
        .arg("--count")
        .write_stdin(input)
        .assert()
        .success()
        .stdout(expected);
    Ok(())
}


fn run_outfile(test: &Test) -> TestResult {
    let expected = fs::read_to_string(test.out)?;
    let outfile = NamedTempFile::new()?;
    let outpath = &outfile.path().to_str().unwrap();

    Command::cargo_bin(PRG)?
        .args(&[test.input, outpath])
        .assert()
        .success()
        .stdout();
    let contents = fs::read_to_string(&outpath)?;
    assert_eq!(&expected, &contents);

    Ok(())
}
