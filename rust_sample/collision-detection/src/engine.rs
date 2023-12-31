use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Mutex;
use std::cell::RefCell;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::channel::mpsc::{UnboundedReceiver, unbounded};
use futures::channel::oneshot::channel;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen::prelude::Closure;
use web_sys::{self, HtmlImageElement, CanvasRenderingContext2d, KeyboardEvent};


use crate::browser;
use crate::schema::{Rect, Point};


enum KeyPress {
    KeyUp(KeyboardEvent),
    KeyDown(KeyboardEvent),
}


pub struct Renderer {
    context: CanvasRenderingContext2d,
}

impl Renderer {
    pub fn clear(&self, rect: &Rect) {
        self.context.clear_rect(
            rect.x.into(),
            rect.y.into(),
            rect.width.into(),
            rect.height.into(),
        );
    }

    pub fn draw_image(&self, image: &HtmlImageElement, frame: &Rect, destination: &Rect) {
        let _ = self.context.draw_image_with_html_image_element_and_sw_and_sh_and_dx_and_dy_and_dw_and_dh(
            &image,
            frame.x.into(),
            frame.y.into(),
            frame.width.into(),
            frame.height.into(),
            destination.x.into(),
            destination.y.into(),
            destination.width.into(),
            destination.height.into(),
        ).expect("Drawing is throwing exceptions! Unrecoverable error.");
    }

    pub fn draw_entire_image(&self, image: &HtmlImageElement, position: &Point) {
        self.context.draw_image_with_html_image_element(
            image,
            position.x.into(),
            position.y.into()
        ).expect("Drawing is throwing exceptions! Unrecoverable error.");
    }
}


pub struct Image {
    element: HtmlImageElement,
    position: Point,
    bounding_box: Rect,
}


impl Image {
    pub fn new(element: HtmlImageElement, position: Point) -> Self {
        let bounding_box = Rect {
            x: position.x.into(),
            y: position.y.into(),
            width: element.width() as f32,
            height: element.height() as f32,
        };
        Self { element, position, bounding_box }
    }

    pub fn draw(&self, renderer: &Renderer) {
        renderer.draw_entire_image(&self.element, &self.position)
    }

    pub fn bounding_box(&self) -> &Rect {
        &self.bounding_box
    }
}


#[async_trait(?Send)]
pub trait Game {
    async fn initialize(&self) -> Result<Box<dyn Game>>;
    fn update(&mut self, keystate: &KeyState);
    fn draw(&self, context: &Renderer);
}

const FRAME_SIZE: f32 = 1.0 / 60.0 * 1000.0;

pub struct GameLoop {
    last_frame: f64,
    accumulate_delta: f32,
}

type SharedLoopClosure = Rc<RefCell<Option<browser::LoopClosure>>>;

impl GameLoop {
    pub async fn start(game: impl Game + 'static) -> Result<()> {
        let mut keyevent_receiver = prepare_input()?;
        let mut game = game.initialize().await?;
        let mut game_loop = GameLoop {
            last_frame: browser::now()?,
            accumulate_delta: 0.0
        };

        let renderer = Renderer {
            context: browser::context()?,
        };

        let f: SharedLoopClosure = Rc::new(RefCell::new(None));
        let g = f.clone();

        let mut keystate = KeyState::new();
        *g.borrow_mut() = Some(browser::create_raf_closure(move |perf: f64| {
            process_input(&mut keystate, &mut keyevent_receiver);
            game_loop.accumulate_delta += (perf - game_loop.last_frame) as f32;
            while game_loop.accumulate_delta > FRAME_SIZE {
                game.update(&keystate);
                game_loop.accumulate_delta -= FRAME_SIZE;
            }
            game_loop.last_frame = perf;

            game.draw(&renderer);
            let _ = browser::request_animation_frame(f.borrow().as_ref().unwrap());
        }));

        browser::request_animation_frame(
            g.borrow().as_ref().ok_or_else(|| anyhow!("GameLoop: Loop is None"))?,
        )?;
        Ok(())
    }
}


pub struct KeyState {
    pressed_keys: HashMap<String, KeyboardEvent>,
}

impl KeyState {
    fn new() -> Self {
        KeyState { pressed_keys: HashMap::new() }
    }

    pub fn is_pressed(&self, code: &str) -> bool {
        self.pressed_keys.contains_key(code)
    }

    pub fn set_pressed(&mut self, code: &str, event: web_sys::KeyboardEvent) {
        self.pressed_keys.insert(code.into(), event);
    }

    pub fn set_released(&mut self, code: &str) {
        self.pressed_keys.remove(code.into());
    }
}


pub async fn load_image(source: &str) -> Result<HtmlImageElement> {
    let image = browser::new_image()?;

    let (complete_tx, complete_rx) = channel::<Result<()>>();
    let success_tx = Rc::new(Mutex::new(Some(complete_tx)));
    let error_tx = Rc::clone(&success_tx);
    let success_callback = browser::closure_once(move || {
        if let Some(success_tx) = success_tx.lock().ok().and_then(|mut opt| opt.take()) {
            let _ = success_tx.send(Ok(()));
        }
    });

    let error_callback: Closure<dyn FnMut(JsValue)> = browser::closure_once(move |error| {
        if let Some(error_tx) = error_tx.lock().ok().and_then(|mut opt| opt.take()) {
            let _ = error_tx.send(Err(anyhow!("Error Loading Image: {:#?}", error)));
        }
    });

    image.set_onload(Some(success_callback.as_ref().unchecked_ref()));
    image.set_onerror(Some(error_callback.as_ref().unchecked_ref()));
    image.set_src(source);

    let _ = complete_rx.await?;

    Ok(image)
}


fn prepare_input() -> Result<UnboundedReceiver<KeyPress>> {
    let (keydown_sender, keyevent_receiver) = unbounded();
    let keydown_sender = Rc::new(RefCell::new(keydown_sender));
    let keyup_sender = Rc::clone(&keydown_sender);

    let onkeydown = browser::closure_wrap(
        Box::new(move |keycode: KeyboardEvent| {
            let _ = keydown_sender.borrow_mut().start_send(KeyPress::KeyDown(keycode));
        }) as Box<dyn FnMut(KeyboardEvent)>
    );

    let onkeyup = browser::closure_wrap(
        Box::new(move |keycode: KeyboardEvent| {
            let _ = keyup_sender.borrow_mut().start_send(KeyPress::KeyUp(keycode));
        }) as Box<dyn FnMut(KeyboardEvent)>
    );

    browser::window()?.set_onkeydown(Some(onkeydown.as_ref().unchecked_ref()));
    browser::window()?.set_onkeyup(Some(onkeyup.as_ref().unchecked_ref()));

    onkeydown.forget();
    onkeyup.forget();

    Ok(keyevent_receiver)
}

fn process_input(state: &mut KeyState, keyevent_receiver: &mut UnboundedReceiver<KeyPress>)
{
    loop {
        match keyevent_receiver.try_next() {
            Ok(None) => break,
            Err(_err) => break,
            Ok(Some(event)) => match event {
                KeyPress::KeyUp(event) => state.set_released(&event.code()),
                KeyPress::KeyDown(event) => state.set_pressed(&event.code(), event),
            }
        };
    }
}