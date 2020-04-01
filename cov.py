#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import *
from tkinter import filedialog
from PIL import Image,ImageTk
import os
from keras.models import load_model
import efficientnet.keras as efn
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as k
import logging
logging.getLogger('tensorflow').disabled = True


# In[2]:


print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
model=load_model(r"Covid.h5")


# In[78]:



def res(x):
    a =cv2.imread(x)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    image=cv2.resize(a,(300,300))
    c=np.expand_dims(image,0)
    return(efn.preprocess_input(c))

def analiz(x): #buradaki fonksiyon resim directory alıyor ve yz ile analiz edip ısı haritalı resmi bulunduğu dosyaya kaydediyor
    
    sonuc=""
    pred2=model.predict(res(x))
    ind=np.argmax(pred2[0])
    if ind==0:
        sonuc="Covid"
    if ind==1:
        sonuc="Normal"
    if ind==2:
        sonuc="Pnemio"
    
    vector=model.output[:,ind]
    last_conv=model.get_layer("top_conv")
    grads=k.gradients(vector,last_conv.output)[0]
    pooled_grad=k.mean(grads,axis=(0,1,2))
    iterate=k.function([model.input],[pooled_grad,last_conv.output[0]])
    pooled_grad_value,conv_layer_value=iterate([res(x)])
    for i in range(1536):
        conv_layer_value[:,:,i] *= pooled_grad_value[i]
    heatmap=np.mean(conv_layer_value,axis=-1)

    plt.rcParams["figure.figsize"]=(16,8)
    img=cv2.imread(x)
    img=cv2.resize(img,(300,300))
    heatmap=np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)
    heatmap=cv2.resize(heatmap,(img.shape[1],img.shape[0]))
    heatmap=np.uint8(255*heatmap)
    i=7
    heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    z=heatmap*0.4+img
    cv2.imwrite('xrayimg.jpg',z)
    
    
    return(sonuc)
    


# In[111]:


img_names=[]
sonuclar1=[]
ivals=[]
def addapp():   # Buradaki fonksiyon dosya eklemek için kullanıyoruz.
    canvas.delete("all")
    
    d.place_forget()
    c.place_forget()
    f.place_forget()
    l.place_forget()
    e.place_forget()
    s.place_forget()
    t.place_forget()
    ivals.extend([ates.get(),oksuruk.get(),temas.get(),nefes.get(),hastalık.get(),e.get()])
    
    filename=filedialog.askopenfilename(initialdir="/", title="Select File",filetypes=(("all files","*.*"),("executables","*.exe")))
    if len(img_names)==1:
        img_names.clear()
    
    img_names.append(filename)

def analiz2():  # daha önce yazılmış analiz fonksiyonunu çalıştırıyoruz ve sonuclar1 listesine ekliyoruz
    sonuclar1.append(analiz(img_names[0]))
    for z in sonuclar1:
        canvas.create_text(250,380,text="AI sonucu:" + z,fill="red")
        canvas.create_text(100,380,text="Semptom Sonucu:Ortalama Riskli")
    path = "xrayimg.jpg"    
    img = ImageTk.PhotoImage(Image.open(path))
    root.img = img
    canvas.create_image(250,200,image=img)
    
    
    if len(sonuclar1)>=1:
        sonuclar1.clear()

def sempc():   #Semptom sayfasını açıyoruz 
    global c,d,f,l,e,s,t
    canvas.delete("all")
    c= Checkbutton(canvas,text="Ateşim 38C üstünde",variable=ates)
    c.place(relx=.5, rely=.1, anchor="c")

    d= Checkbutton(canvas,text="Öksürüğüm var",variable=oksuruk)
    d.place(relx=.5, rely=.2, anchor="c")

    s= Checkbutton(canvas,text="Nefes darlığım var ",variable=nefes)
    s.place(relx=.5, rely=.42, anchor="c")
    
    f= Checkbutton(canvas,text="Aşağıdaki hastalık veya durumlardan herhangi birine sahibim \n-İmmun yetmezlik \n-Hipertansiyon \n-Kemoterapi \n-Diyabet\n-Astım  ",variable=hastalık)
    f.place(relx=.45, rely=.57, anchor="c")
    
    t= Checkbutton(canvas,text="COVID-19 için test yapılmış ve sonucu pozitif gelmiş kişilerle temasım oldu",variable=temas)
    t.place(relx=.45, rely=.28, anchor="c")
    
    e = Entry(canvas,text="")
    e.place(relx=.5, rely=.7, anchor="c")

    l = Label(canvas,text="Yaşınızı giriniz:")
    l.place(relx=.2, rely=.7, anchor="c")
    

    
    


# In[112]:


root = Tk()
root.configure(background='purple')
root.title("Covid AI")
root.iconbitmap('ico1.ico')

canvas = Canvas(root,height=500,width=500)
canvas.pack()

ates = IntVar()
oksuruk=IntVar()
temas=IntVar()
nefes=IntVar()
hastalık=IntVar()







yol = "ai.jpg"    
imgz = ImageTk.PhotoImage(Image.open(yol))
canvas.create_image(25,150,image=imgz)
#frame = Frame(root,bg="white")
#frame.place(relwidth=0.8,relheight=0.8,relx=0.1,rely=0.1)

openfile = Button(root,text="Xray görüntüsü seçiniz",padx=20,pady=15,command=addapp,bg="black",fg="white")
openfile.place(relx=.5, rely=.85, anchor="c")

run = Button(root,text="Analiz",padx=20,pady=15,command=analiz2,bg="orange",fg="red")
run.place(relx=.80, rely=.85, anchor="c")

semp = Button(root,text="Semptom",padx=20,pady=15,command=sempc,bg="orange",fg="blue")
semp.place(relx=.20,rely=.85, anchor="c")

#entry=Entry(text="Yaşınızı giriniz")
#entry.place(relx=0.3,rely=.3)
#guess = entry.get()
 
root.mainloop()


# In[ ]:




