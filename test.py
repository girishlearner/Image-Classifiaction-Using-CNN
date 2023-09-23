import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
#load the trained model to classify the images
from keras.models import load_model
model = load_model('ImageClassification.h5')
#dictionary to label all the CIFAR-10 dataset classes.
classes = {
    0:'Aeroplane',
    1:'Automobile',
    2:'Bird',
    3:'Cat',
    4:'Deer',
    5:'Dog',
    6:'Frog',
    7:'Horse',
    8:'Ship',   
    9:'Truck'
}
#initialise GUI
top=tk.Tk()
top.geometry('1920x1080')
top.title('Image Classification Using CNN')
top.configure(background='#090979')
label=Label(top,background='#CDCDCD', font=('times New Roman',27,'bold'))
sign_image = Label(top)
def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((32,32))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    pred=numpy.argmax(model.predict([image])[0],axis=-1)
    sign = classes[pred]
    print(sign)
    label.configure(background = '#090979',foreground='#ffffff', text=sign)
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#000000', foreground='white',font=('arial',20,'bold'))
    classify_b.place(relx=0.79,rely=0.46)
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass
upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#000000', foreground='white',font=('arial',20,'bold'))
upload.pack(side=BOTTOM,pady=80)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Image Classification Using CNN",pady=20, font=('times new roman',50,'bold'))
heading.configure(background='#090979',foreground='#ffffff')
heading.pack()
top.mainloop()
