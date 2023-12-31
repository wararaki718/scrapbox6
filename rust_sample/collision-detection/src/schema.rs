use std::collections::HashMap;
use serde::Deserialize;


#[derive(Deserialize)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn intersects(&self, rect: &Rect) -> bool {
        self.x < (rect.x + rect.width) && (self.x + self.width) > rect.x && self.y < (rect.y + rect.height) && (self.y + self.height) > rect.y
    }
}


#[derive(Deserialize, Clone)]
pub struct SheetRect {
    pub x: i16,
    pub y: i16,
    pub w: i16,
    pub h: i16,
}


#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Cell {
    pub frame: SheetRect,
    pub sprite_source_size: SheetRect,
}


#[derive(Deserialize, Clone)]
pub struct Sheet {
    pub frames: HashMap<String, Cell>
}


#[derive(Clone, Copy)]
pub struct Point {
    pub x: i16,
    pub y: i16,
}
