use wasm_bindgen::prelude::*;
use web_sys;
use wasm_bindgen::JsCast;

// This is like the `main` function, except for JavaScript.
#[wasm_bindgen(start)]
pub fn main_js() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    // Your code goes here!
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap().dyn_into::<web_sys::HtmlCanvasElement>().unwrap();

    let context = canvas.get_context("2d").unwrap().unwrap().dyn_into::<web_sys::CanvasRenderingContext2d>().unwrap();
    
    context.move_to(300.0, 0.0);
    context.begin_path();
    context.line_to(0.0, 600.0);
    context.line_to(600.0, 600.0);
    context.line_to(300.0, 0.0);
    context.close_path();
    context.stroke();
    context.fill();

    Ok(())
}
