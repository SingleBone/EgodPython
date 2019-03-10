import tkinter as tk
import pickle
import tkinter.messagebox

window = tk.Tk()
window.title('欢迎登陆知问空间！')
window.geometry('450x300')

canvas = tk.Canvas(window,height=200,width=500)
img = tk.PhotoImage(file='./welcome.gif')
image = canvas.create_image(0,0,anchor='nw',image=img)
canvas.pack()

tk.Label(window,text='用户名:').place(x=50,y=150)
var_usr_name = tk.StringVar()
var_usr_name.set('example@yingcai.com')
tk.Entry(window,textvariable=var_usr_name).place(x=160,y=150)
tk.Label(window,text='密码:').place(x=50,y=190)
var_usr_password = tk.StringVar()
tk.Entry(window,textvariable=var_usr_password,show='*').place(x=160,y=190)

def usr_login():
    usr_name = var_usr_name.get()
    usr_password = var_usr_password.get()
    
    try:
        with open('usr_info.pickle','rb') as usr_file:
            usr_info = pickle.load(usr_file)
    except FileNotFoundError:
        with open('usr_info.pickle','wb') as usr_file:
            usr_info = {'admin':'admin'}
            pickle.dump(usr_info,usr_file)
            
    if usr_name in usr_info:
        if usr_password == usr_info[usr_name]:
            tk.messagebox.showinfo(title='Welcom!',message='How are you? '+usr_name)
        else:
            tk.messagebox.showerror(message='Sorry, your password is wrong, try again!')
    else:
        is_sign_up = tk.messagebox.askyesno(title='Welcome!',message='You have not sign up yet, '+\
                                           ' would you sign up now?')
        if is_sign_up:
            usr_signup()

def usr_signup():
    def signup():
        new_name = var_usr_name_sign.get()
        new_password = var_usr_password_sign.get()
        new_password_confirm = var_usr_password_sign2.get()
        
        try:
            with open('usr_info.pickle','rb') as usr_file:
                exist_usr_info = pickle.load(usr_file)
        except FileNotFoundError:
            with open('usr_info.pickle','wb') as usr_file:
                usr_info = {'admin':'admin'}
                pickle.dump(usr_info,usr_file)
        
        if new_password != new_password_confirm:
            tk.messagebox.showerror(message='Password and Comfirm password must be the same!')
        elif new_name in exist_usr_info:
            tk.messagebox.showerror(message='The user name already exists!')
        else:
            exist_usr_info[new_name]=new_password
            with open('usr_info.pickle','wb') as usr_file:
                pickle.dump(exist_usr_info,usr_file)
            tk.messagebox.showinfo('Welcome!','You have successfully signed up!')
            window_signup.destroy()
    # sign up window
    window_signup = tk.Toplevel(window)
    window_signup.geometry('350x200')
    window_signup.title('Sign up now')
    
    tk.Label(window_signup,text='Name:').place(x=40,y=30)
    tk.Label(window_signup,text='Password:').place(x=40,y=60)
    tk.Label(window_signup,text='Comfirm password:').place(x=40,y=90)
    var_usr_name_sign = tk.StringVar()
    var_usr_password_sign = tk.StringVar()
    var_usr_password_sign2 = tk.StringVar()
    tk.Entry(window_signup,textvariable=var_usr_name_sign).place(x=160,y=30)
    tk.Entry(window_signup,textvariable=var_usr_password_sign,show='*').place(x=160,y=60)
    tk.Entry(window_signup,textvariable=var_usr_password_sign2,show='*').place(x=160,y=90)
    tk.Button(window_signup,text='Sign up!',command=signup).place(x=100,y=130)
    tk.Button(window_signup,text='Cancel',command=window_signup.destroy).place(x=200,y=130)

    

tk.Button(window,text='登陆',command=usr_login).place(x=160,y=230)
tk.Button(window,text='注册',command=usr_signup).place(x=230,y=230)

window.mainloop()