#!/usr/bin/env python

import wx

length = 600

class Graph:
    
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.graph = dict()

        for x in range(0, grid_size):
            for y in range(0, grid_size):
                node = (x, y)
                self.graph[node] = list() 
            
    
    

class DrawPanel(wx.Panel):
    
    def __init__(self, parent, idx = -1, grid_size = 8):
        super(DrawPanel, self).__init__(parent, idx)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.grid_size = grid_size

    def OnPaint(self, event=None):
        dc = wx.PaintDC(self)
        dc.Clear()
        
        rect = self.GetClientRect()
        side = min(rect.width, rect.height)
        side = 0.9 * side;
 
        dc.SetPen(wx.Pen(wx.BLACK, 2))

        rect_startx = (rect.width - side) / 2;
        rect_starty = (rect.height - side) / 2;
       
        dc.DrawRectangle(rect_startx, rect_starty, side, side)

        cell_side = side / self.grid_size

        for i in range(0, self.grid_size):
            dc.DrawLine(rect_startx, rect_starty + i*cell_side, rect_startx + side, rect_starty + i*cell_side)
            dc.DrawLine(rect_startx + i*cell_side, rect_starty, rect_startx + i*cell_side, rect_starty + side)

class GraphWindow(wx.Frame):

    def __init__(self, title):
        wx.Frame.__init__(self, None, title=title, pos=(150,150), size=(length, length))

        mainSizer = wx.BoxSizer(wx.VERTICAL)

        panelSizer = wx.BoxSizer(wx.VERTICAL)
        self.panel = DrawPanel(self, -1)
        panelSizer.Add(self.panel , 1, wx.EXPAND)

        mainSizer.Add(panelSizer, 8, wx.EXPAND)        
        self.SetSizer(mainSizer)

if __name__ == "__main__":
    app = wx.App()
    frame = GraphWindow("Prim's algorithm")
    frame.Show()
    app.MainLoop() 

