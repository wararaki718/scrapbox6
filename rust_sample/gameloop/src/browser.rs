use std::future::Future;
use wasm_bindgen_futures::JsFuture;
use web_sys::{self, Window, Document, HtmlCanvasElement, CanvasRenderingContext2d, Response, HtmlImageElement};
use wasm_bindgen::{JsCast, JsValue, closure::{Closure, WasmClosureFnOnce}};
use anyhow::{anyhow, Result};

macro_rules! log {
    ($( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}

pub fn window() -> Result<Window> {
    web_sys::window().ok_or_else(|| anyhow!("No Window Found"))
}

pub fn document() -> Result<Document> {
    window()?.document().ok_or_else(|| anyhow!("No Document Found"))
}

pub fn canvas() -> Result<HtmlCanvasElement> {
    document()?
        .get_element_by_id("canvas")
        .ok_or_else(|| anyhow!("No Canvas Element found with ID 'canvas'"))?
        .dyn_into::<HtmlCanvasElement>()
        .map_err(|element| anyhow!("Error converting {:#?} to HtmlCanvasElement", element))
}

pub fn context() -> Result<CanvasRenderingContext2d> {
    canvas()?
        .get_context("2d")
        .map_err(|js_value| anyhow!("Error getting 2d context {:#?}", js_value))?
        .ok_or_else(|| anyhow!("No 2d context found"))?
        .dyn_into::<CanvasRenderingContext2d>()
        .map_err(|element| {anyhow!("Error converting {:#?} to CanvasRenderingContext2d", element)})
}

pub fn spawn_local<F>(future: F)
where
    F: Future<Output = ()> + 'static,
{
    wasm_bindgen_futures::spawn_local(future)
}

pub async fn fetch_with_str(resource: &str) -> Result<JsValue> {
    JsFuture::from(window()?.fetch_with_str(resource))
        .await
        .map_err(|error| anyhow!("error fetching {:#?}", error))
}

pub async fn fetch_json(json_path: &str) -> Result<JsValue> {
    let response_value = fetch_with_str(json_path).await?;
    let response: Response = response_value.dyn_into().map_err(|element| anyhow!("Error converting {:#?} to Response", element))?;

    JsFuture::from(
        response.json().map_err(|error| anyhow!("Could not get JSON from response {:#?}", error))?
    ).await.map_err(|error| anyhow!("error fetching JSON {:#?}", error))
}

pub fn new_image() -> Result<HtmlImageElement> {
    HtmlImageElement::new()
        .map_err(|error| anyhow!("Could not create HtmlImageElement: {:#?}", error))
}

pub fn closure_once<F, A, R>(fn_once: F) -> Closure<F::FnMut>
where
    F: 'static + WasmClosureFnOnce<A, R>,
{
    Closure::once(fn_once)
}

