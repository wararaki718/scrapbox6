#[macro_use]
mod browser;
mod schema;
mod engine;
mod game;
mod segment;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;

use game::WalkTheDog;
use engine::GameLoop;
use gloo_utils::format::JsValueSerdeExt;

// This is like the `main` function, except for JavaScript.
#[wasm_bindgen(start)]
pub fn main_js() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    browser::spawn_local(async move {
        let game = WalkTheDog::new();

        GameLoop::start(game).await.expect("Could not start game loop");
    });


    Ok(())
}
