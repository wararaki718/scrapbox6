use std::rc::Rc;
use std::sync::Mutex;

use wasm_bindgen::prelude::*;
use web_sys;
use gloo_utils::format::JsValueSerdeExt;

pub mod schema;
use crate::schema::Sheet;


async fn fetch_json(json_path: &str) -> Result<JsValue, JsValue> {
    let window = web_sys::window().unwrap();
    let response_value = wasm_bindgen_futures::JsFuture::from(
        window.fetch_with_str(json_path)
    ).await?;
    let response: web_sys::Response = response_value.dyn_into()?;

    wasm_bindgen_futures::JsFuture::from(response.json()?).await
}


#[wasm_bindgen(start)]
pub fn main_js() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap().dyn_into::<web_sys::HtmlCanvasElement>().unwrap();

    let context = canvas.get_context("2d").unwrap().unwrap().dyn_into::<web_sys::CanvasRenderingContext2d>().unwrap();

    wasm_bindgen_futures::spawn_local(async move {
        let json = fetch_json("rhb.json").await.expect("Could not fetch rhb.json");
        let sheet: Sheet = json.into_serde().expect("Could not convert rhb.json into a sheet structure");

        let (success_tx, success_rx) = futures::channel::oneshot::channel::<Result<(), JsValue>>();
        let success_tx = Rc::new(Mutex::new(Some(success_tx)));
        let error_tx = Rc::clone(&success_tx);
        let image = web_sys::HtmlImageElement::new().unwrap();

        let callback = Closure::once(move || {
            if let Some(success_tx) = success_tx.lock().ok().and_then(|mut opt| opt.take()){
                let _ = success_tx.send(Ok(()));
            }
            web_sys::console::log_1(&JsValue::from_str("loaded"));
        });

        let error_callback = Closure::once(move |error| {
            if let Some(error_tx) = error_tx.lock().ok().and_then(|mut opt| opt.take()) {
                let _ = error_tx.send(Err(error));
            }
        });
        image.set_onload(Some(callback.as_ref().unchecked_ref()));
        image.set_onerror(Some(error_callback.as_ref().unchecked_ref()));
        image.set_src("rhb.png");
        let _ = success_rx.await;

        //let _ = context.draw_image_with_html_image_element(&image, 0.0, 0.0);
        let sprite = sheet.frames.get("Run (1).png").expect("Cell not found");
        let _ = context.draw_image_with_html_image_element_and_sw_and_sh_and_dx_and_dy_and_dw_and_dh(
            &image,
            sprite.frame.x.into(),
            sprite.frame.y.into(),
            sprite.frame.w.into(),
            sprite.frame.h.into(),
            300.0,
            300.0,
            sprite.frame.w.into(),
            sprite.frame.h.into(),
        );
    });

    Ok(())
}
