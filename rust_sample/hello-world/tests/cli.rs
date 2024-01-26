use std::process::Command;
use assert_cmd::Command as AssertCommand;
use assert_cmd::cargo::CommandCargoExt;
use assert_cmd::assert::OutputAssertExt;

#[test]
fn works() {
    assert!(true)
}


#[test]
fn runs() {
    let mut cmd = Command::new("ls");
    let res = cmd.output();
    assert!(res.is_ok());
}


#[test]
fn runs_assert() {
    let mut cmd = AssertCommand::cargo_bin("hello-world").unwrap();
    cmd.assert().success();
}


#[test]
fn runs_stdout() {
    let mut cmd = Command::cargo_bin("hello-world").unwrap();
    cmd.assert().success().stdout("Hello, world!\n");
}
