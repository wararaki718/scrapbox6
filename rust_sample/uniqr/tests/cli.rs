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
    out: "tests/expected/empty.txt.out",
    out_count: "tests/expected/empty.txt.c.out",
};

const ONE: Test = Test {
    input: "tests/inputs/one.txt",
    out: "tests/expected/one.txt.out",
    out_count: "tests/expected/one.txt.c.out",
};

const TWO: Test = Test {
    input: "tests/inputs/two.txt",
    out: "tests/expected/two.txt.out",
    out_count: "tests/expected/two.txt.c.out",
};

const THREE: Test = Test {
    input: "tests/inputs/three.txt",
    out: "tests/expected/three.txt.out",
    out_count: "tests/expected/three.txt.c.out",
};

const SKIP: Test = Test {
    input: "tests/inputs/skip.txt",
    out: "tests/expected/skip.txt.out",
    out_count: "tests/expected/skip.txt.c.out",
};


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
fn dies_bad_file() -> TestResult {
    let bad = gen_bad_file();
    let expected = format!("{}: .* [(]os error 2[)]", bad);
    Command::cargo_bin(PRG)?
        .arg(bad)
        .assert()
        .failure()
        .stderr(predicate::str::is_match(expected)?);
    Ok(())
}


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


fn run_stdin(test: &Test) -> TestResult {
    let input = fs::read_to_string(test.input)?;
    let expected = fs::read_to_string(test.out)?;
    Command::cargo_bin(PRG)?
        .write_stdin(input)
        .assert()
        .success()
        .stdout(expected);
    Ok(())
}


fn run_stdin_count(test: &Test) -> TestResult {
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
        .stdout("");
    let contents = fs::read_to_string(&outpath)?;
    assert_eq!(&expected, &contents);

    Ok(())
}

fn run_outfile_count(test: &Test) -> TestResult {
    let expected = fs::read_to_string(test.out_count)?;
    let outfile = NamedTempFile::new()?;
    let outpath = &outfile.path().to_str().unwrap();

    Command::cargo_bin(PRG)?
        .args(&[test.input, outpath, "--count"])
        .assert()
        .success()
        .stdout("");
    let contents = fs::read_to_string(&outpath)?;
    assert_eq!(&expected, &contents);

    Ok(())
}

fn run_stdin_outfile_count(test: &Test) -> TestResult {
    let input = fs::read_to_string(test.input)?;
    let expected = fs::read_to_string(test.out_count)?;
    let outfile = NamedTempFile::new()?;
    let outpath = &outfile.path().to_str().unwrap();

    Command::cargo_bin(PRG)?
        .args(&["-", outpath, "--count"])
        .write_stdin(input)
        .assert()
        .success()
        .stdout("");
    let contents = fs::read_to_string(&outpath)?;
    assert_eq!(&expected, &contents);

    Ok(())
}

#[test]
fn empty() -> TestResult {
    run(&EMPTY)
}