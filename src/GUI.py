import tkinter
from tkinter import *
from backend import Analyze
from datetime import date
from svm_realtime import predict_svm
from nb_realtime import predict_nb
from nn_realtime import predict_nn
import sqlite3

conn = sqlite3.connect('dtab.db')
#conn.execute("create table User(User_Id int, Name text);")
#conn.execute("create table Post(Posted_Text text not null, Posting_Date date, Polarity real);")

root = Tk()
root.title('Sentiment Analysis using Dictionary Method')

text = StringVar()
sen_lex = StringVar()
sen_nb = StringVar()
sen_svm = StringVar()
sen_nn = StringVar()
username = StringVar()
user_id = StringVar()
time_yy = StringVar()
time_mm = StringVar()
time_dd = StringVar()

def Calculate(event):
    t =text.get()
    print(t)

    o1 = Analyze(t)
    sen_lex.set(o1)
    print(sen_lex.get())

    o2 = predict_svm(t)
    sen_svm.set(o2)
    print(sen_svm.get())

    o3 = predict_nb(t)
    sen_nb.set(o2)
    print(sen_nb.get())

    o4 = predict_nn(t)
    sen_nn.set(o4)
    print(sen_nn.get())

def Insert(event):
    User_Id = user_id.get()
    Name = username.get()
    Posted_Text = text.get()
    dd = int(time_dd.get())
    mm = int(time_mm.get())
    yy = int(time_yy.get())
    Posting_date = date(yy, mm, dd)
    Polarity = float(sen_lex.get())
    print(User_Id, Name, Posted_Text, Posting_date)

    conn.execute("insert into User(User_Id, Name) values(?, ?)", (User_Id, Name))
    conn.execute("insert into Post(Posted_Text, Posting_Date, Polarity) values (?, ?, ?)",
                 (Posted_Text, Posting_date, Polarity))
    conn.commit()

def Exit(event):
    conn.close()
    tkinter.quit()

L1 = Label(root, text='Enter text:')
L1.grid(row=1, column=0)
E1= Entry(root, textvariable=text)
E1.grid(row=1, column=1)

L2 = Label(root, text="Lexical Sentiment Polarity:")
L2.grid(row=2, column=0)
L3 = Label(root, textvariable=sen_lex)
L3.grid(row=2, column=1)

L10 = Label(root, text="NB Sentiment Polarity:")
L10.grid(row=3, column=0)
L11 = Label(root, textvariable=sen_nb)
L11.grid(row=3, column=1)

L12 = Label(root, text="SVM Sentiment Polarity:")
L12.grid(row=4, column=0)
L13 = Label(root, textvariable=sen_svm)
L13.grid(row=4, column=1)

L14 = Label(root, text="NN Sentiment Polarity:")
L14.grid(row=5, column=0)
L15 = Label(root, textvariable=sen_nn)
L15.grid(row=5, column=1)

B1 = Button(root, text='Calculate')
B1.grid(row=2, column=3)
B1.bind('<Button-1>', Calculate)

L4 = Label(root, text="Username:")
L4.grid(row=6, column=0)
E2 = Entry(root, textvariable=username)
E2.grid(row=6, column=1)

L5 = Label(root, text="User_ID:")
L5.grid(row=7, column=0)
E3 = Entry(root, textvariable=user_id)
E3.grid(row=7, column=1)

L6 = Label(root, text="YYYY")
L6.grid(row=8, column=0)
E4 = Entry(root, textvariable=time_yy)
E4.grid(row=8, column=1)
L8 = Label(root, text="MM")
L8.grid(row=8, column=2)
E5 = Entry(root, textvariable=time_mm)
E5.grid(row=8, column=3)
L9 = Label(root, text="DD")
L9.grid(row=8, column=4)
E6 = Entry(root, textvariable=time_dd)
E6.grid(row=8, column=5)

B1 = Button(root, text='Insert')
B1.grid(row=9, column=3)
B1.bind('<Button-1>', Insert)

B2 = Button(root, text='Exit', command=root.destroy)
B2.grid(row=9, column=4)
B2.bind('<Button-1>', Exit)


root.mainloop()