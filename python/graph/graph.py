#!/usr/bin/env python

import wx

length = 600

class Graph:
    
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.adj_list = dict()

        for x in range(0, grid_size):
            for y in range(0, grid_size):
                node = (x, y)
                self.adj_list[node] = list() 
        
        for node in self.adj_list.keys():
            x, y = node
            neighbours = list()

            if x > 0:
                neighbours.append((x - 1, y))

            if x < grid_size - 1:
                neighbours.append((x + 1, y))

            if y > 0:
                neighbours.append((x, y - 1))

            if y < grid_size - 1:
                neighbours.append((x, y + 1))
            
            self.adj_list[node] = neighbours
    

class DrawPanel(wx.Panel):
    
    def __init__(self, parent, idx = -1, grid_size = 8):
        super(DrawPanel, self).__init__(parent, idx)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.grid_size = grid_size
        self.graph = Graph(self.grid_size)

    def get_dimensions(self):
        rect = self.GetClientRect()
        side = min(rect.width, rect.height)
        side = 0.9 * side;
 
        startx = (rect.width - side) / 2;
        starty = (rect.height - side) / 2;

        return startx, starty, side

    def draw_edge(self, node1, node2, cell_side):
        dc = wx.PaintDC(self)
 
        x1, y1 = node1
        x2, y2 = node2

        startx, starty, side = self.get_dimensions() 
 
        if x1 == x2 and y1 == y2:
            return
        
        if abs(x1 - x2) > 1 or abs(y1 - y2) > 1:
            return

        if abs(x1 - x2) == 1:
            x = max(x1, x2)
            lx1 = startx + x*cell_side
            ly1 = starty + y1*cell_side
            lx2 = startx + x*cell_side
            ly2 = starty + (y1 + 1)*cell_side
            dc.DrawLine(lx1, ly1, lx2, ly2)

        if abs(y1 - y2) == 1:
            y = max(y1, y2)
            lx1 = startx + x1*cell_side
            ly1 = starty + y*cell_side
            lx2 = startx + (x1 + 1)*cell_side
            ly2 = starty + y*cell_side
            dc.DrawLine(lx1, ly1, lx2, ly2)

    def OnPaint(self, event=None):
        dc = wx.PaintDC(self)
        dc.Clear()
        dc.SetPen(wx.Pen(wx.BLACK, 2))

        startx, starty, side = self.get_dimensions() 
        dc.DrawRectangle(startx, starty, side, side)

        cell_side = side / self.grid_size

        for node in self.graph.adj_list.keys():
            x, y = node
            neighbours = self.graph.adj_list[node]
            for neighbour in neighbours:
                self.draw_edge(node, neighbour, cell_side)


class GraphWindow(wx.Frame):

    def __init__(self, title):
        wx.Frame.__init__(self, None, title=title, pos=(150,150), size=(length, length))

        mainSizer = wx.BoxSizer(wx.VERTICAL)

        panelSizer = wx.BoxSizer(wx.VERTICAL)
        self.panel = DrawPanel(self, -1, grid_size = 4)
        panelSizer.Add(self.panel , 1, wx.EXPAND)

        mainSizer.Add(panelSizer, 8, wx.EXPAND)        
        self.SetSizer(mainSizer)

if __name__ == "__main__":
    app = wx.App()
    frame = GraphWindow("Prim's algorithm")
    frame.Show()
    app.MainLoop() 

