import H4BL_NN as network
import numpy as np
from tkinter import *

training_inputs = np.array([[1, 1, 1, 1],
                            [1, 0, 1, 0],
                            [1, 1, 0, 1],
                            [1, 0, 0, 0],
                            [0, 1, 1, 1],
                            [0, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 0, 0, 0]])

training_outputs = np.array([[0, 1],
                            [1, 0],
                            [0, 1],
                            [0, 0],
                            [1, 0],
                            [1, 0],
                            [0, 1],
                            [0, 0]])

def output():
    data = [inp1.get(), inp2.get(), inp3.get(), inp4.get()]

    label4.config(text=data)

    erg = neural_network.think(np.array([int(inp1.get()),
                                   int(inp2.get()),
                                   int(inp3.get()),
                                   int(inp4.get())]))

    label5.config(text=erg)

    inp1.delete(0, END)
    inp2.delete(0, END)
    inp3.delete(0, END)
    inp4.delete(0, END)

    if erg[0]<erg[1]:
        print("Drehe nach rechts!")
        showBut.config(text="Rechts")
        showBut.grid(column=1, row=7, pady=5)
    else:
        print("Drehe nach links!")
        showBut.config(text="Links")
        showBut.grid(column=0, row=7, pady=5)

neural_network = network.NeuralNetwork()

in1 = StringVar
in2 = StringVar
in3 = StringVar
in4 = StringVar

root = Tk()
root.title("H4BL NeuralNet")
root.geometry("350x300")

label1 = Label(root, text="Input 1", width=25)
label2 = Label(root, text="Input 2", width=25)
label3 = Label(root, text="Input 3", width=25)
label6 = Label(root, text="Input 4", width=25)
inp1 = Entry(root, textvariable=in1, width=20)
inp2 = Entry(root, textvariable=in2, width=20)
inp3 = Entry(root, textvariable=in3, width=20)
inp4 = Entry(root, textvariable=in4, width=20)
but = Button(root, width=20, command=output, text="Berechnen")
label4 = Label(root, text="???", width=25)
label5 = Label(root, text="???", width=25)
showBut = Button(root, width=15, height=5, bg="green")

label1.grid(column=0, row=0)
inp1.grid(column=0, row=1)
label2.grid(column=1, row=0)
inp2.grid(column=1, row=1)
label3.grid(column=0, row=2)
inp3.grid(column=0, row=3)
label6.grid(column=1, row=2)
inp4.grid(column=1, row=3)
but.grid(column=0, row=4, pady=20)
label4.grid(column=0, row=5)
label5.grid(column=0, row=6)

neural_network.train(training_inputs, training_outputs, 50000)
root.mainloop()

