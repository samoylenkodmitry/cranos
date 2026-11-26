use cranos_boot_info::{FramebufferInfo, PixelFormat};

const CURSOR_W: usize = 12;
const CURSOR_H: usize = 16;

// 1 = black outline, 2 = white fill, 0 = transparent
const CURSOR_MASK: [[u8; CURSOR_W]; CURSOR_H] = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0],
    [1, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0],
    [1, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0],
    [1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0],
    [1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0],
    [1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0],
    [1, 2, 2, 1, 2, 2, 1, 0, 0, 0, 0, 0],
    [1, 2, 1, 0, 1, 2, 2, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 2, 2, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
];

pub struct Framebuffer {
    ptr: *mut u32,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: PixelFormat,
    cursor_drawn: bool,
    cursor_x: u32,
    cursor_y: u32,
    saved_under: [u32; CURSOR_W * CURSOR_H],
}

impl Framebuffer {
    pub fn new(info: FramebufferInfo) -> Self {
        Self {
            ptr: info.phys_addr as *mut u32,
            width: info.width,
            height: info.height,
            stride: info.stride_pixels,
            format: info.format,
            cursor_drawn: false,
            cursor_x: info.width / 2,
            cursor_y: info.height / 2,
            saved_under: [0; CURSOR_W * CURSOR_H],
        }
    }

    #[inline]
    pub fn pack_rgb(&self, r: u8, g: u8, b: u8) -> u32 {
        match self.format {
            PixelFormat::Bgrx8 => (b as u32) | ((g as u32) << 8) | ((r as u32) << 16),
            PixelFormat::Rgbx8 => (r as u32) | ((g as u32) << 8) | ((b as u32) << 16),
        }
    }

    pub fn hide_cursor(&mut self) {
        if !self.cursor_drawn {
            return;
        }
        let cx = self.cursor_x as usize;
        let cy = self.cursor_y as usize;
        for row in 0..CURSOR_H {
            let y = cy + row;
            if y >= self.height as usize {
                break;
            }
            for col in 0..CURSOR_W {
                let x = cx + col;
                if x >= self.width as usize {
                    break;
                }
                if CURSOR_MASK[row][col] != 0 {
                    let px = self.saved_under[row * CURSOR_W + col];
                    unsafe {
                        self.ptr.add(y * self.stride as usize + x).write(px);
                    }
                }
            }
        }
        self.cursor_drawn = false;
    }

    pub fn draw_cursor(&mut self, x: u32, y: u32) {
        self.hide_cursor();
        self.cursor_x = x.min(self.width.saturating_sub(1));
        self.cursor_y = y.min(self.height.saturating_sub(1));
        let cx = self.cursor_x as usize;
        let cy = self.cursor_y as usize;
        let black = self.pack_rgb(8, 12, 20);
        let white = self.pack_rgb(248, 250, 252);

        for row in 0..CURSOR_H {
            let py = cy + row;
            if py >= self.height as usize {
                break;
            }
            for col in 0..CURSOR_W {
                let px = cx + col;
                if px >= self.width as usize {
                    break;
                }
                let m = CURSOR_MASK[row][col];
                if m != 0 {
                    let p = unsafe { self.ptr.add(py * self.stride as usize + px) };
                    self.saved_under[row * CURSOR_W + col] = unsafe { p.read() };
                    unsafe {
                        p.write(if m == 1 { black } else { white });
                    }
                }
            }
        }
        self.cursor_drawn = true;
    }

    /// Blits a row-major straight-alpha RGBA8 rectangle (as rendered by
    /// Cranpose `PixelsRenderer::draw_picture_area`) into the linear GOP
    /// framebuffer. Uses sequential writes so PAT Write-Combining batches full
    /// 64-byte cache-line bursts over PCIe/display memory.
    pub fn blit_rgba_rect(&mut self, x: u32, y: u32, w: u32, h: u32, rgba: &[u8]) {
        if w == 0 || h == 0 || x >= self.width || y >= self.height {
            return;
        }
        let copy_w = w.min(self.width - x) as usize;
        let copy_h = h.min(self.height - y) as usize;
        let src_stride = w as usize * 4;
        let bgrx = self.format == PixelFormat::Bgrx8;

        for row in 0..copy_h {
            let src_row = &rgba[row * src_stride..row * src_stride + copy_w * 4];
            let dst_row = unsafe {
                self.ptr
                    .add((y as usize + row) * self.stride as usize + x as usize)
            };
            if bgrx {
                for col in 0..copy_w {
                    let i = col * 4;
                    let px = (src_row[i + 2] as u32)
                        | ((src_row[i + 1] as u32) << 8)
                        | ((src_row[i] as u32) << 16);
                    unsafe { dst_row.add(col).write(px) };
                }
            } else {
                for col in 0..copy_w {
                    let i = col * 4;
                    let px = (src_row[i] as u32)
                        | ((src_row[i + 1] as u32) << 8)
                        | ((src_row[i + 2] as u32) << 16);
                    unsafe { dst_row.add(col).write(px) };
                }
            }
        }
    }
}
