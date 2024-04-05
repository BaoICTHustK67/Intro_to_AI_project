from tkinter import *
import pickle


def Build_GUI(model, tfidf_vectorizer, filename):
    pickle.dump(model, open(filename, 'wb'))
    spam_model = pickle.load(open("finalized_model.sav",'rb'))
    
    def check_spam():
        text = spam_text_Entry.get()
        is_spam = spam_model.predict(tfidf_vectorizer.transform([text]))
        if is_spam == 1:
            print("text is spam")
            my_string_var.set("Result: text is spam")
        else:
            print("text is not spam")
            my_string_var.set("Result: text is not spam")


    win = Tk()
    
    win.geometry("400x600")
    win.configure(background="white")
    win.title("Message/Comment Spam Detector")
    
    title = Label(win, text="Message/Comment Spam Detector", bg="gray",
                  width="300",height="2",fg="white",
                  font=("Calibri 20 bold italic underline")).pack()
    
    spam_text = Label(win, text="Enter your Text: ",bg="cyan",
                       font=("Verdana 12")).place(x=12,y=100)
    spam_text_Entry = Entry(win, textvariable=spam_text,width=33)
    spam_text_Entry.place(x=155, y=105)
    
    my_string_var = StringVar()
    my_string_var.set("Result: ")
    
    
    print_spam = Label(win, textvariable=my_string_var,bg="cyan",
                        font=("Verdana 12")).place(x=12,y=200)

    button = Button(win, text="Submit",width="12",height="1",activebackground="red",bg="Pink",command=check_spam,font=("Verdana 12")).place(x=12,y=150)
    
    
    win.mainloop()