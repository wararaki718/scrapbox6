use clap::{App, Arg};
use std::error::Error;

type MyResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug)]
pub struct Config {
    files: Vec<String>,
    lines: usize,
    bytes: Option<usize>,
}

pub fn get_args() -> MyResult<Config> {
    let matches = App::new("headr")
        .version("0.1.0")
        .author("wararaki")
        .about("Rust head")
        .arg(
            Arg::with_name("files")
                .value_name("FILE")
                .help("Input file(s)")
                .multiple(true)
                .default_value("-"),
        )
        .arg(
            Arg::with_name("lines")
                .long("lines")
                .short("n")
                .help("print the first K lines")
                .default_value("10")
                .takes_value(true)
                .conflicts_with("bytes"),
        )
        .arg(
            Arg::with_name("bytes")
                .long("bytes")
                .short("c")
                .takes_value(true)
                .help("print the first K bytes")
        )
        .get_matches();

    // let files = matches.values_of_lossy("files").unwrap();
    // let lines = matches.values_of_lossy("lines").unwrap();
    // let bytes = matches.values_of_lossy("bytes").unwrap();

    Ok(Config{
        files: vec!["-".to_string()],
        lines: 10,
        bytes: None,
    })
}


pub fn run(config: Config) -> MyResult<()> {
    println!("{:#?}", config);
    Ok(())
}


fn parse_positive_int(val: &str) -> MyResult<usize> {
    match val.parse() {
        Ok(n) if n > 0 => Ok(n),
        _ => Err(From::from(val)),
    }
}


#[test]
fn test_parse_positive_int() {
    let res = parse_positive_int("3");
    assert!(res.is_ok());
    assert_eq!(res.unwrap(), 3);

    let res = parse_positive_int("foo");
    assert!(res.is_err());
    assert_eq!(res.unwrap_err().to_string(), "foo".to_string());

    let res = parse_positive_int("0");
    assert!(res.is_err());
    assert_eq!(res.unwrap_err().to_string(), "0".to_string());
}