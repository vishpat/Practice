#!/usr/bin/env python

class svgshape:
    
    def gcode(self):
        pass

    def svg(self):
        pass

class rect(svgshape):
   
    def __init__(self, x = 0, y = 0, 
                width = 0, height = 0, rx = 0, ry = 0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rx = rx
        self.ry = ry

class circle(svgshape):
        
    def __init__(self, cx = 0, cy = 0, r = 0):
        self.cx = cx
        self.cy = cy 
        self.r = r 
 
class ellipse(svgshape):
        
    def __init__(self, cx = 0, cy = 0, rx = 0, ry = 0):
        self.cx = cx 
        self.cy = cy 
        self.rx = rx
        self.ry = ry
    
