use std::collections::HashMap;
use serde::Deserialize;


#[derive(Deserialize)]
pub struct Rect {
    pub x: u16,
    pub y: u16,
    pub w: u16,
    pub h: u16,
}


#[derive(Deserialize)]
pub struct Cell {
    pub frame: Rect,
}


#[derive(Deserialize)]
pub struct Sheet {
    pub frames: HashMap<String, Cell>
}
