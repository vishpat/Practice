#!/usr/bin/env python

import logging
import xml.etree.ElementTree as ET
import simplepath

class svgpath(object):
    
    def __init__(self, xml):
        self.xml = xml.lower()
        self.xml_tree = None
        try:
            self.xml_tree = ET.fromstring(xml)
        except:
            logging.error("Unable to parse xml %s", xml)

    def path(self):
        raise NotImplementedError

    def path_tag(self, p):
        return "<path d=\"" + p + "\"/>"

    def gcode(self):
        raise NotImplementedError

    def __str__(self):
        return self.xml        

class rect(svgpath):
  
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
            logging.error("rect: Unable to get the attributes for %s", self.xml)

    def path(self):
        a = list()
        a.append( ['M ', [self.x, self.y]] )
        a.append( [' l ', [self.width, 0]] )
        a.append( [' l ', [0, self.height]] )
        a.append( [' l ', [-self.width, 0]] )
        a.append( [' Z', []] )
        return self.path_tag(simplepath.formatPath(a))

class ellipse(svgpath):

    def __init__(self, xml):
        super(ellipse, self).__init__(xml)

        if (not (self.xml_tree == None) and self.xml_tree.tag == 'ellipse'):
            ellipse_el = self.xml_tree
            self.cx  = int(ellipse_el.get('cx')) if ellipse_el.get('cx') else 0
            self.cy  = int(ellipse_el.get('cy')) if ellipse_el.get('cy') else 0
            self.rx = int(ellipse_el.get('rx')) if ellipse_el.get('rx') else 0
            self.ry = int(ellipse_el.get('ry')) if ellipse_el.get('ry') else 0
        else:
            self.cx = self.cy = self.rx = self.ry = 0
            logging.error("ellipse: Unable to get the attributes for %s", self.xml)

    def path(self):
        x1 = self.cx - self.rx
        x2 = self.cx + self.rx
        p = 'M %f,%f ' % ( x1, self.cy ) + \
            'A %f,%f ' % ( self.rx, self.ry ) + \
            '0 1 0 %f,%f ' % ( x2, self.cy ) + \
            'A %f,%f ' % ( self.rx, self.ry ) + \
            '0 1 0 %f,%f' % ( x1, self.cy )
        return self.path_tag(p)

class circle(ellipse):
    def __init__(self, xml):
        super(ellipse, self).__init__(xml)

        if (not (self.xml_tree == None) and self.xml_tree.tag == 'circle'):
            circle_el = self.xml_tree
            self.cx  = int(circle_el.get('cx')) if circle_el.get('cx') else 0
            self.cy  = int(circle_el.get('cy')) if circle_el.get('cy') else 0
            self.rx = int(circle_el.get('r')) if circle_el.get('r') else 0
            self.ry = self.rx
        else:
            self.cx = self.cy = self.r = 0
            logging.error("Circle: Unable to get the attributes for %s", self.xml)

class line(svgpath):

    def __init__(self, xml):
        super(line, self).__init__(xml)

        if (not (self.xml_tree == None) and self.xml_tree.tag == 'line'):
            line_el = self.xml_tree
            self.x1  = int(line_el.get('x1')) if line_el.get('x1') else 0
            self.y1  = int(line_el.get('y1')) if line_el.get('y1') else 0
            self.x2 = int(line_el.get('x2')) if line_el.get('x2') else 0
            self.y2 = int(line_el.get('y2')) if line_el.get('y2') else 0
        else:
            self.x1 = self.y1 = self.x2 = self.y2 = 0
            logging.error("line: Unable to get the attributes for %s", self.xml)

    def path(self):
        a = []
        a.append( ['M ', [self.x1, self.y1]] )
        a.append( ['L ', [self.x2, self.y2]] )
        return self.path_tag(simplepath.formatPath(a))

class polycommon(svgpath):

    def __init__(self, xml, polytype):
        super(polycommon, self).__init__(xml)
        self.points = list()

        if (not (self.xml_tree == None) and self.xml_tree.tag == polytype):
            polycommon_el = self.xml_tree
            points = int(polycommon_el.get('points')) if polycommon_el.get('points') else 0
            for pa in points.split():
                points.append(pa)
        else:
            logging.error("polycommon: Unable to get the attributes for %s", self.xml)


class polygon(polycommon):

    def __init__(self, xml):
         super(polygon, self).__init__(xml, 'ploygon')

    def path(self):
        d = "M " + self.points[0]
        for i in range( 1, len(self.points) ):
            d += " L " + self.points[i]
        d += " Z"
        return self.path_tag(d)

class polyline(polycommon):

    def __init__(self, xml):
         super(polyline, self).__init__(xml, 'ployline')

    def path(self):
        d = "M " + self.points[0]
        for i in range( 1, len(self.points) ):
            d += " L " + self.points[i]
        return self.path_tag(d)

if __name__ == "__main__":
    svg_rect = """<rect x="1" y="1" width="200" height="300"/>""" 
    r = rect(svg_rect)
    assert r.path() == """<path d="M 1 1 l 200 0 l 0 300 l -200 0 Z"/>"""

    svg_line = """<line x1="0" y1="0" x2="200" y2="200"/>"""
    l = line(svg_line)
    print svg_line, l.path()

    svg_circle = """<circle cx="100" cy="50" r="40"/>"""
    c = circle(svg_circle)
    assert c.path() == """<path d="M 60.000000,50.000000 A 40.000000,40.000000 0 1 0 140.000000,50.000000 A 40.000000,40.000000 0 1 0 60.000000,50.000000"/>"""

