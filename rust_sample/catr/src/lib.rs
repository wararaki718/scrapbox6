use std::error::Error;
use clap::{App, Arg};
use std::fs::File;
use std::io::{self, BufRead, BufReader};

#[derive(Debug)]
pub struct Config {
    files: Vec<String>,
    number_lines: bool,
    number_nonblank_lines: bool,
}

type MyResult<T> = Result<T, Box<dyn Error>>;

fn open(filename: &str) -> MyResult<Box<dyn BufRead>> {
    match filename {
        "-" => Ok(Box::new(BufReader::new(io::stdin()))),
        _ => Ok(Box::new(BufReader::new(File::open(filename)?))),
    }
}

pub fn run(config: Config) -> MyResult<()> {
    for filename in config.files {
        match open(&filename) {
            Err(err) => eprintln!("Failed to open {}: {}", filename, err),
            Ok(reader) => {
                let mut nonblank = 1;
                for (i, line) in reader.lines().enumerate() {
                    let result = line?;
                    if config.number_lines {
                        println!("{:6}\t{}", i+1, result);
                    }
                    else if config.number_nonblank_lines{
                        if result.is_empty() {
                            println!("");
                        } else {
                            println!("{:6}\t{}", nonblank, result);
                            nonblank += 1;
                        }
                    } 
                    else {
                        println!("{}", result);
                    }
                }
            },
        }
    }
    Ok(())
}

pub fn get_args() -> MyResult<Config> {
    let matches = App::new("catr")
        .version("0.1.0")
        .author("wararaki")
        .about("Rust cat")
        .arg(
            Arg::with_name("files")
                .value_name("FILE")
                .help("Input file(s)")
                .multiple(true)
                .default_value("-"),
        )
        .arg(
            Arg::with_name("number")
                .long("number")
                .short("n")
                .help("Number lines")
                .takes_value(false)
                .conflicts_with("number_nonblank"),
        )
        .arg(
            Arg::with_name("number_nonblank")
                .long("number-nonblank")
                .short("b")
                .help("Number nonblank lines")
                .takes_value(false),
        )
        .get_matches();

    let files = matches.values_of_lossy("files").unwrap();
    let number_lines = matches.is_present("number");
    let number_nonblank = matches.is_present("number_nonblank");

    Ok(Config{
        files: files,
        number_lines: number_lines,
        number_nonblank_lines: number_nonblank,
    })
}
