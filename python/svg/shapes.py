#!/usr/bin/env python

import logging
import xml.etree.ElementTree as ET

class svgshape(object):
    
    def __init__(self, xml):
        self.xml = xml.lower()
        self.xml_tree = None
        try:
            self.xml_tree = ET.fromstring(xml)
        except:
            logging.error("Unable to parse xml %s", xml)

    def gcode(self):
        raise NotImplementedError

    def __str__(self):
        return self.xml        

class rect(svgshape):
  
    def __init__(self, xml):
        super(rect, self).__init__(xml)

        if (not (self.xml_tree == None) and self.xml_tree.tag == 'rect'):
            rect_el = self.xml_tree
            self.x  = int(rect_el.get('x')) if rect_el.get('x') else 0
            self.y  = int(rect_el.get('y')) if rect_el.get('y') else 0
            self.rx = int(rect_el.get('rx')) if rect_el.get('rx') else 0
            self.ry = int(rect_el.get('ry')) if rect_el.get('ry') else 0
            self.width = int(rect_el.get('width')) if rect_el.get('width') else 0
            self.height = int(rect_el.get('height')) if rect_el.get('height') else 0
        else:
            self.x = self.y = self.rx = self.ry = self.width = self.height = 0
            logging.error("Unable to get the attributes for %s", self.xml)
 

    def __str__(self):
        return "<rect x=%d y=%d width=%d height=%d/>" % (self.x, self.y, self.width, self.height)

class circle(svgshape):
    pass    

class ellipse(svgshape):
    pass        

if __name__ == "__main__":
    r = rect("""<rect x="1" y="1" width="1198" height="398"/>""")
    print r

