use std::fs::File;
use std::io::{BufReader, BufRead};
use std::path::Path;


fn main() -> std::io::Result<()> {
    let path = Path::new("hello.txt");
    let display = path.display();

    let file = File::open(&path).unwrap();

    let reader = BufReader::new(file);
    
    for (i, line) in reader.lines().enumerate() {
        println!("{}: {}", i, line.unwrap());
    }
    println!("DONE!");
    Ok(())
}
