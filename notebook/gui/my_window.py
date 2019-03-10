import tkinter as tk
# Initial the window
window = tk.Tk()
window.title('Who love Eve?')
window.geometry('700x500')
# Claim the functions
def print_selection():
    value = lb.get(lb.curselection())
    var1.set(value)
def insert_point():
    var = e.get()
    t.insert('insert',var)
def add_end():
    var = e.get()
    t.insert('end',var)
# Claim the variables 
var1 = tk.StringVar()
var2 = tk.StringVar()
var2.set((11,22,33,44))
lb = tk.Listbox(window,listvariable=var2)
list_items = [1,2,3,4]
for item in list_items:
    lb.insert('end',item)
lb.insert(1,'first')
lb.insert(2,'second')
l = tk.Label(window,bg='yellow',height=4,textvariable=var1)
b1 = tk.Button(window,text='insert point',height=5,command=insert_point)
b2 = tk.Button(window,text='add point at end',width = 15,height=2,command=add_end)
b3 = tk.Button(window,text='print selection',width=15,height=2,command=print_selection)
e = tk.Entry(window,show="*")
t = tk.Text(window,height=3)
# pack all the variables
l.pack()
b3.pack()
lb.pack()
b1.pack()
b2.pack()
e.pack()
t.pack()
# mainloop
window.mainloop()