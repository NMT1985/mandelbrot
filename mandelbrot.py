import numpy as np
import pylab as pl
from timeit import default_timer as timer
from numba import njit
import math

@njit
def draw_image(image, start_x, start_y, pixel_size, max_iters, bailout):
    height, width = image.shape[0], image.shape[1]
    for y in range(height):
        cy = start_y + y * pixel_size
        for x in range(width):
            cx = start_x + x * pixel_size

            zx,zy, radius2 = 0.0, 0.0, 0.0
            iteration = 0
            while iteration < max_iters and radius2 <= bailout:
                zx, zy = zx*zx - zy*zy + cx, 2*zx*zy + cy
                radius2 = zx*zx + zy*zy
                iteration += 1

            if iteration<max_iters:
                alpha = iteration*0.05
                alpha = (alpha%1)*255
                image[height-y-1, x, 0] = alpha
                image[height-y-1, x, 1] = alpha
                image[height-y-1, x, 2] = alpha

    return

def draw(width, height, center_real=0.0, center_imag=0.0, magnification=1.0, max_iters=100, bailout=4.0):

    pixel_size = 4/magnification / width
    start_x,  start_y = center_real-pixel_size*width/2, center_imag-pixel_size*height/2
    image = np.zeros((height, width,3), dtype = np.uint8)

    start = timer()
    draw_image(image, start_x, start_y, pixel_size,  max_iters, bailout)
    duration = timer() - start

    return image, duration

# dummy call for JIT
# image, duration = draw(16,16)
# print(f"JIT compiling in {duration:.5f} sec.")

width, height = 640, 480
center_real, center_imag = -0.5, 0.0
magnification = 1.2
max_iters = 1000
#draw
image, duration = draw(width, height, center_real,center_imag, magnification, max_iters)
print(f"{width}x{height} pixels / {duration:.3f} sec.")

pl.figure(dpi=200)
pl.imshow(image)
pl.axis("off")

pl.show()
from PIL import Image
Image.fromarray(image).save('Mandelbrot(Simple).png')