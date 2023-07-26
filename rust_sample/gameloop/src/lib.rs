#[macro_use]
mod browser;
mod schema;
mod engine;

use wasm_bindgen::prelude::*;
use gloo_utils::format::JsValueSerdeExt;

// This is like the `main` function, except for JavaScript.
#[wasm_bindgen(start)]
pub fn main_js() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let context = browser::context().expect("Counld not get browser context");
    browser::spawn_local(async move {
        let sheet: schema::Sheet = browser::fetch_json("rhb.json")
            .await
            .expect("Could not fetch rhb.json")
            .into_serde()
            .expect("Could not convert rhb.json into a Sheet structure");

        let image = engine::load_image("rhb.png")
            .await
            .expect("Cound not load rhb.png");

        let mut frame = -1;
    });

    let document = browser::document().expect("No Document Found");


    Ok(())
}
