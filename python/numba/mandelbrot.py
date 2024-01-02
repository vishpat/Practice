from PIL import Image
from numba import njit, prange,jit
import numpy as np
from timeit import default_timer as timer

#frame parameters
width = 1000 #pixels
x = -0.65
y = 0
xRange = 3.4
aspectRatio = 4/3 

precision = 500

height = round(width / aspectRatio)
yRange = xRange / aspectRatio
minX = x - xRange / 2
maxX = x + xRange / 2
minY = y - yRange / 2
maxY = y + yRange / 2

img = Image.new('RGB', (width, height), color = 'black')
pixels = img.load()
pixels = np.zeros((width, height), dtype=np.uint32)


@njit
def hsv2rgb(h, s, v):
    
    """HSV to RGB
    
    :param float h: 0.0 - 360.0
    :param float s: 0.0 - 1.0
    :param float v: 0.0 - 1.0
    :return: rgb 
    :rtype: list
    
    """
    
    c = v * s
    x = c * (1 - abs(((h/60.0) % 2) - 1))
    m = v - c
    
    if 0.0 <= h < 60:
        rgb = (c, x, 0)
    elif 0.0 <= h < 120:
        rgb = (x, c, 0)
    elif 0.0 <= h < 180:
        rgb = (0, c, x)
    elif 0.0 <= h < 240:
        rgb = (0, x, c)
    elif 0.0 <= h < 300:
        rgb = (x, 0, c)
    elif 0.0 <= h < 360:
        rgb = (c, 0, x)
        
    r = int((rgb[0] + m) * 255)
    g = int((rgb[1] + m) * 255)
    b = int((rgb[2] + m) * 255)

    return r << 16 | g << 8 | b


@njit(parallel=True)
def generate_image(pixels):
  
  def pixelColor(row, col):

    def powerColor(distance, exp, const, scale):
        color = distance**exp
        return hsv2rgb(const + scale * color,1 - 0.6 * color,0.9)

    x = minX + col * xRange / width
    y = maxY - row * yRange / height
    oldX = x
    oldY = y
    for i in range(precision + 1):
        a = x*x - y*y #real component of z^2
        b = 2 * x * y #imaginary component of z^2
        x = a + oldX #real component of new z
        y = b + oldY #imaginary component of new z
        if x*x + y*y > 4:
            break
    if i < precision:
        distance = (i + 1) / (precision + 1)
        rgb = powerColor(distance, 0.2, 0.27, 1.0)
        return rgb
    
    return 0

  for row in prange(height):
      for col in prange(width):
          color = pixelColor(row, col)
          pixels[col,row] = color >> 16 & 255 |  color >> 8 & 255 | color & 255 

start = timer()
generate_image(pixels)
end = timer()
print(end - start)
img = Image.fromarray(pixels, 'RGB')
img.save('output.png')
