import json
import numpy as np
from sklearn.utils import shuffle



steps = 250
numDrawings = 1000
classNames=["airplane","apple","bird","book","bridge","bus","car","cat","chair","clock",
            "computer","diamond","dog","ear","eyeglasses","fish","flower","guitar","harp","hot air balloon",
            "hourglass","house","key","leaf","lightning","moon","mug","octopus","pants","pencil",
            "pizza","rainbow","rifle","sailboat","scissors","shovel","skyscraper","snake","snowflake","strawberry",
            "sun","sword","television","toothbrush","tree","trumpet","t-shirt","umbrella","violin","wine glass"] #50 classes

data = [] #shape: (50*numDrawings,maxSteps,5)
labels = [] #50*numDrawings


def convertToVector(line, maxSteps):#takes in 1 drawing (1 line in the ndjson file), return vector array of stroke arrays
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

    if len(strokes)>maxSteps:
        return None
    seq = np.array(strokes, dtype=np.float32)
    padding = np.zeros((maxSteps- len(seq),5), dtype=np.float32)#pad zeroes, remember mask during training !
    return np.concatenate([seq, padding], axis=0)


def loadData(numDrawings): #
    for i in range (len(classNames)):
        className = classNames[i]
        count = 0

        with open(f"RNN/data/{className}.ndjson", "r") as f:
            for line in f:
                if count>=numDrawings:
                    break
                try:
                    dic = json.loads(line)
                    vec = convertToVector(dic,steps)
                    if vec is not None:
                        data.append(vec)
                        labels.append(i)
                        count+=1
                except json.JSONDecodeError:
                    continue  #skip bad lines
    
    dataNp = np.array(data, dtype=np.float32)
    labelsNp = np.array(labels)
    dataNp, labelsNp = shuffle(dataNp, labelsNp, random_state=42)


    #Normalization : NOT zero mean normalization (only divide mby std)
    all_deltas = np.concatenate([dataNp[:, :, 0].flatten(), dataNp[:, :, 1].flatten()]) #all deltax and delta y in 1d list
    scale = np.std(all_deltas)
    dataNp[:, :, 0] /= scale
    dataNp[:, :, 1] /= scale

    return dataNp,labelsNp,scale

dataNp, labelsNp, scale= loadData(numDrawings)
np.savez_compressed("processed_data.npz",data=dataNp,labels=labelsNp,scale=scale)

 
                
