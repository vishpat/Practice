use ndarray::{Array2, Zip};

const WIDTH: u32 = 64*512;
const HEIGHT: u32 = WIDTH;
const X: f64 = -0.65;
const Y: f64 = 0.0;
const X_RANGE: f64 = 3.4;
const Y_RANGE: f64 = X_RANGE;

const PRECISION: u32 = 500;

const MIN_X: f64 = (X - X_RANGE) / 2.0;
const MAX_Y: f64 = (Y + Y_RANGE) / 2.0;

fn pixel_color(row: u32, col: u32) -> u32 {
    let mut x = MIN_X + col as f64 * X_RANGE / WIDTH as f64;
    let mut y = MAX_Y - row as f64 * Y_RANGE / HEIGHT as f64;

    let old_x = x;
    let old_y = y;

    for _ in 0..PRECISION {
        let a = x * x - y * y;
        let b = 2.0 * x * y;
        x = a + old_x; 
        y = b + old_y;
        if x * x + y * y > 4.0 {
            return 0x000000;
        }
    }

    0xffffff
}

fn main() {
    let mut pixels = Array2::<u32>::zeros((HEIGHT as usize, WIDTH as usize));
    Zip::indexed(pixels.view_mut())
        .par_for_each(|(row, col), pixel| {
            *pixel = pixel_color(row as u32, col as u32);
        }); 
    
    let img = image::ImageBuffer::from_fn(WIDTH, HEIGHT, |col, row| {
        let pixel = pixels[[row as usize, col as usize]];
        image::Rgb([(pixel >> 16) as u8, (pixel >> 8) as u8, pixel as u8])
    });
    img.save("mandelbrot.png").unwrap();
}
