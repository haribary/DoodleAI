import json
import numpy as np



maxStrokes = 250

def convertToVector(line, maxStrokes):#takes in 1 drawing (1 line in the ndjson file), return vector array of strokes
    #[deltaX,deltaY,pen down, pen up, end of drawing]
    if not line.get("recognized",True):
        return None
    drawing = line["drawing"]

    strokes = []
    for stroke in drawing:
        Xpoints,Ypoints = stroke
        prevX,prevY = Xpoints[0],Ypoints[0]
        strokes.append([0,0,1,0,0])

        for i in range (1,len(Xpoints)):
            x,y = Xpoints[i],Ypoints[i]
            dx,dy = x-prevX, y-prevY
            strokes.append([dx,dy,1,0,0])
            prevX,prevY = x,y

        strokes[-1][2] = 0
        strokes[-1][3] = 1 #end of stroke

    strokes[-1][2] = 0  
    strokes[-1][3] = 0  
    strokes[-1][4] = 1  #end of drawing

    if len(strokes)>maxStrokes:
        return None
    return np.array(strokes)


