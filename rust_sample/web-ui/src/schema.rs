use std::collections::HashMap;
use serde::Deserialize;
use web_sys::{HtmlImageElement, AudioContext, AudioBuffer};

use anyhow::{Result, Ok};
use crate::engine::Renderer;
use crate::{sound, browser};

#[derive(Clone, Copy, Default)]
pub struct Point {
    pub x: i16,
    pub y: i16,
}

#[derive(Default)]
pub struct Rect {
    pub position: Point,
    pub width: i16,
    pub height: i16,
}

impl Rect {
    pub fn new(position: Point, width: i16, height: i16) -> Self {
        Rect {
            position,
            width,
            height,
        }
    }

    pub fn new_from_x_y(x: i16, y: i16, width: i16, height: i16) -> Self {
        Rect::new(
            Point { x, y },
            width,
            height,
        )
    }

    pub fn x(&self) -> i16 {
        self.position.x
    }

    pub fn y(&self) -> i16 {
        self.position.y
    }

    pub fn intersects(&self, rect: &Rect) -> bool {
        self.x() < rect.right() && self.right() > rect.x() && self.y() < rect.bottom() && self.bottom() > rect.y()
    }

    pub fn right(&self) -> i16 {
        self.x() + self.width
    }

    pub fn bottom(&self) -> i16 {
        self.y() + self.height
    }

    pub fn set_x(&mut self, x: i16) {
        self.position.x = x;
    }

    pub fn set_y(&mut self, y: i16) {
        self.position.y = y;
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

pub struct SpriteSheet {
    sheet: Sheet,
    image: HtmlImageElement,
}

impl SpriteSheet {
    pub fn new(sheet: Sheet, image: HtmlImageElement) -> Self {
        SpriteSheet { sheet, image }
    }

    pub fn cell(&self, name: &str) -> Option<&Cell> {
        self.sheet.frames.get(name)
    }

    pub fn draw(&self, renderer: &Renderer, source: &Rect, destination: &Rect) {
        renderer.draw_image(&self.image, source, destination);
    }
}



#[derive(Clone)]
pub struct Sound {
    buffer: AudioBuffer,
}



#[derive(Clone)]
pub struct Audio {
    context: AudioContext,
}

impl Audio {
    pub fn new() -> Result<Self> {
        Ok(
            Audio {
                context: sound::create_audio_context()?,
            }
        )
    }

    pub async fn load_sound(&self, filename: &str) -> Result<Sound> {
        let array_buffer = browser::fetch_array_buffer(filename).await?;
        let audio_buffer = sound::decode_audio_data(&self.context, &array_buffer).await?;
        Ok(
            Sound {
                buffer: audio_buffer,
            }
        )
    }

    pub fn play_sound(&self, sound: &Sound) -> Result<()> {
        sound::play_sound(&self.context, &sound.buffer, sound::LOOPING::NO)
    }

    pub fn play_looping_sound(&self, sound: &Sound) -> Result<()> {
        sound::play_sound(&self.context, &sound.buffer, sound::LOOPING::YES)
    }
}
