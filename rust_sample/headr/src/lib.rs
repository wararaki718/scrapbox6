use clap::{App, Arg};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::io::Read;

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
                .conflicts_with("bytes")
                .value_name("LINES")
        )
        .arg(
            Arg::with_name("bytes")
                .long("bytes")
                .short("c")
                .takes_value(true)
                .help("print the first K bytes")
                .value_name("BYTES")
        )
        .get_matches();

    let files = matches.values_of_lossy("files").unwrap();
    let lines = matches.value_of("lines")
        .map(parse_positive_int)
        .transpose()
        .map_err(|e| format!("illegal line count =={}", e))?;
    let bytes = matches.value_of("bytes")
        .map(parse_positive_int)
        .transpose()
        .map_err(|e| format!("illegal byte count =={}", e))?;

    Ok(Config{
        files: files,
        lines: lines.unwrap(),
        bytes,
    })
}


fn open(filename: &str) -> MyResult<Box<dyn BufRead>> {
    match filename {
        "-" => Ok(Box::new(BufReader::new(io::stdin()))),
        _ => Ok(Box::new(BufReader::new(File::open(filename)?))),
    }
}


pub fn run(config: Config) -> MyResult<()> {
    let n_files = config.files.len();

    for (i, filename) in config.files.iter().enumerate() {
        match open(&filename) {
            Err(err) => eprintln!("{}: {}", filename, err),
            Ok(mut file) => {
                if n_files > 1 {
                    println!(
                        "{}==> {} <==",
                        if i > 0 { "\n" } else { "" },
                        filename
                    );
                }

                if let Some(n_bytes) = config.bytes {
                    let mut handle = file.take(n_bytes as u64);
                    let mut buffer = vec![0; n_bytes];
                    let bytes_read = handle.read(&mut buffer)?;
                    print!("{}", String::from_utf8_lossy(&buffer[..bytes_read]));
                } else {
                    let mut line = String::new();
                    for _ in 0..config.lines {
                        let bytes = file.read_line(&mut line)?;
                        if bytes == 0 {
                            break;
                        }
                        print!("{}", line);
                        line.clear();
                    }
                }
            }
        };
    }

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