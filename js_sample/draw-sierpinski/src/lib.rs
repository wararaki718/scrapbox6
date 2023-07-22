use rand::prelude::*;
use wasm_bindgen::prelude::*;
use web_sys;
use wasm_bindgen::JsCast;


fn draw_triangle(context: &web_sys::CanvasRenderingContext2d, points: [(f64, f64); 3], color: (u8, u8, u8))
{
    let color_str = format!("rgb({}, {}, {})", color.0, color.1, color.2);
    context.set_fill_style(&wasm_bindgen::JsValue::from_str(&color_str));

    let [top, left, right] = points;
    context.move_to(top.0, top.1);
    context.begin_path();
    context.line_to(left.0, left.1);
    context.line_to(right.0, right.1);
    context.line_to(top.0, top.1);
    context.close_path();
    context.stroke();
    context.fill();
}

fn midpoint(point1: (f64, f64), point2: (f64, f64)) -> (f64, f64)
{
    ((point1.0 + point2.0) / 2.0, (point1.1 + point2.1) / 2.0)
}

fn draw_sierpinski(context: &web_sys::CanvasRenderingContext2d, points: [(f64, f64); 3], depth: u8, color: (u8, u8, u8))
{
    draw_triangle(&context, points, color);
    
    let _depth = depth - 1;
    let [top, left, right] = points;

    if _depth > 0 {
        let mut rng = thread_rng();
        let next_color = (
            rng.gen_range(0..255),
            rng.gen_range(0..255),
            rng.gen_range(0..255)
        );

        let left_middle = midpoint(top, left);
        let right_middle = midpoint(top, right);
        let bottom_middle = midpoint(left, right);
        draw_sierpinski(&context, [top, left_middle, right_middle], _depth, next_color);
        draw_sierpinski(&context, [left_middle, left, bottom_middle], _depth, next_color);
        draw_sierpinski(&context, [right_middle, bottom_middle, right], _depth, next_color);
    }
}

// This is like the `main` function, except for JavaScript.
#[wasm_bindgen(start)]
pub fn main_js() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    // Your code goes here!
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap().dyn_into::<web_sys::HtmlCanvasElement>().unwrap();

    let context = canvas.get_context("2d").unwrap().unwrap().dyn_into::<web_sys::CanvasRenderingContext2d>().unwrap();

    draw_sierpinski(&context, [(300.0, 0.0), (0.0, 600.0), (600.0, 600.0)], 5, (0, 255, 0));

    Ok(())
}
