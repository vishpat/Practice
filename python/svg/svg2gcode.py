#!/usr/bin/env python

from xml.dom import minidom
import sys

import shapes

svg_shapes = ['rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'path']

def parse_svg():
    dom = minidom.parse(sys.stdin)
    
    for svg_shape in svg_shapes:
        paths = dom.getElementsByTagName(svg_shape)
        for path in paths:
            shape_class = getattr(shapes, path.nodeName)
            shape_obj = shape_class(path.toxml())
            print shape_obj.svg_path()
    dom.unlink()

if __name__ == "__main__":
    parse_svg()



