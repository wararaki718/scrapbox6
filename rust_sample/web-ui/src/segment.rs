use std::rc::Rc;

use web_sys::HtmlImageElement;

use crate::{game::{Obstacle, Barrier, Platform}, engine::Image, schema::{Point, Rect, SpriteSheet}};

const INITIAL_STONE_OFFSET: i16 = 150;
const STONE_ON_GROUND: i16 = 546;
const FIRST_PLATFORM: i16 = 370;
const LOW_PLATFORM: i16 = 420;
const HIGH_PLATFORM: i16 = 375;

pub fn stone_and_platform(
    stone: HtmlImageElement,
    sprite_sheet: Rc<SpriteSheet>,
    offset_x: i16,
) -> Vec<Box<dyn Obstacle>> {
    vec![
        Box::new(Barrier::new(Image::new(
            stone,
            Point {
                x: offset_x + INITIAL_STONE_OFFSET,
                y: STONE_ON_GROUND,
            }
        ))),
        Box::new(create_floating_platform(
            sprite_sheet,
            Point {
                x: offset_x + FIRST_PLATFORM,
                y: LOW_PLATFORM,
            }
        )),
    ]
}

pub fn platform_and_stone(
    stone: HtmlImageElement,
    sprite_sheet: Rc<SpriteSheet>,
    offset_x: i16,
) -> Vec<Box<dyn Obstacle>> {
    vec![
        Box::new(create_floating_platform(
            sprite_sheet,
            Point {
                x: offset_x + FIRST_PLATFORM,
                y: HIGH_PLATFORM,
            }
        )),
        Box::new(Barrier::new(Image::new(
            stone,
            Point {
                x: offset_x + INITIAL_STONE_OFFSET,
                y: STONE_ON_GROUND,
            }
        ))),
    ]
}


fn create_floating_platform(sprite_sheet: Rc<SpriteSheet>, position: Point) -> Platform {
    Platform::new(
        sprite_sheet,
        position,
        &["13.png", "14.png", "15.png"],
        &[
            Rect::new_from_x_y(0, 0, 60, 54),
            Rect::new_from_x_y(60, 0, 384 - (60 * 2), 93),
            Rect::new_from_x_y(384 - 60, 0, 60, 54)
        ],
    )
}