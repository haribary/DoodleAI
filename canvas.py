import tkinter as tk
from predictImage import predict
from PIL import Image, ImageGrab
import numpy as np
import matplotlib.pyplot as plt
root = tk.Tk()
root.title("Test app")
root.geometry("500x500")



label = tk.Label(root,text=" ")
label.pack(pady=20,padx=20)






def draw(event):
    x,y = event.x, event.y
    canvas.create_oval(x,y,x+15,y+15,fill="white",outline='white')

canvas = tk.Canvas(root,width=280,height=280,bg="black")
canvas.pack()
canvas.bind("<B1-Motion>", draw)#b1motion is draw when left moutse button 




def predictCanvas():
    canvas.update()
    #get position of canvas
    x = (root.winfo_rootx() + canvas.winfo_x())*1.5
    y = (root.winfo_rooty() + canvas.winfo_y())*1.5
    x1 = x + 280*1.5
    y1 = y + 280*1.5
    #crop,grayscale,resize,convert
    img = ImageGrab.grab().crop((x, y, x1, y1)).convert("L")

    # plt.imshow(img, cmap="gray")
    # plt.axis("off") 
    # plt.show()

    img = img.resize((28,28))
    img = np.array(img)

    

    result = predict(img) #list of 3 tuples (category,% probability)
    label.config(text="Top 3 probabilities:\n"
                 + result[0][0] + " - " + str(result[0][1]) +"%"+ "\n"
                 + result[1][0] + " - " + str(result[1][1]) +"%"+ "\n"
                 + result[2][0] + " - " + str(result[2][1])+"%")
    
    

def clearCanvas():
    canvas.delete("all")
    label.config(text="")


predictbutton = tk.Button(root,text="predict",command=predictCanvas)
predictbutton.pack()
clearbutton = tk.Button(root,text="clear",command=clearCanvas)
clearbutton.pack()


root.mainloop()
