import tkinter as tk
from tkinter import *
import datetime
import os
from info_and_check import save_info, check_login
from camfeeds import update_camera_feed, update_overlay_feed


# Global variables

user_info = {'name': '', 'last_name': '', 'username': '', 'password': ''}


log_text = None

# Main window
def window():
    global windows
    windows = Tk()
    windows.geometry("600x400")
    windows.title("Window")
    windows.configure(bg="#ffffff")

    label2 = Label(windows, text="Login or create an account.")
    label2.pack()

    add_button = Button(windows, text='Create an Account', command=create)
    add_button.pack()

    login_button = Button(windows, text='Login', command=login)
    login_button.pack()

    exit_button = Button(windows, text='Save and Exit', command=windows.destroy)
    exit_button.pack()

    windows.mainloop()

# Create account window
def create():
    global creates
    global windows
    global name_entry, last_entry, user_entry, passw_entry
    windows.destroy()

    creates = Tk()
    creates.geometry("600x400")
    creates.title("Create an Account")
    creates.configure(bg='white')

    name = Label(creates, text="Name: ")
    name.pack()

    name_entry = Entry(creates)
    name_entry.pack()

    last = Label(creates, text="Last Name: ")
    last.pack()

    last_entry = Entry(creates)
    last_entry.pack()

    user = Label(creates, text="User: ")
    user.pack()

    user_entry = Entry(creates)
    user_entry.pack()

    passw = Label(creates, text="Password: ")
    passw.pack()

    passw_entry = Entry(creates, show='*')
    passw_entry.pack()

    save_button = Button(creates, text='Save and Exit', command=lambda: [save_info(creates,user_info,name_entry, last_entry, user_entry, passw_entry),window()])
    save_button.pack()

    creates.mainloop()



# Login window
def login():
    global logins
    global windows
    global userlogin_entry, passwlogin_entry
    windows.destroy()
    global root
    logins = Tk()
    logins.geometry("600x400")
    logins.title("Log In")

    user = Label(logins, text="User: ")
    user.pack()

    userlogin_entry = Entry(logins)
    userlogin_entry.pack()

    passw = Label(logins, text="Password: ")
    passw.pack()

    passwlogin_entry = Entry(logins, show='*')
    passwlogin_entry.pack()

    login_button = Button(logins, text='Log In', command=lambda: check_login(logins,userlogin_entry, passwlogin_entry))
    login_button.pack()



# Logged-in window
def loggedin(username, password, key, name):

    global log_text, camera_feed_label, root

    root = tk.Tk()
    root.geometry("600x600")
    root.title("Logged In")
    root.resizable(False, False)
    top_left_frame = tk.Frame(root, bg="white", width=300, height=300)
    top_right_frame = tk.Frame(root, bg="turquoise", width=300, height=300)
    bottom_left_frame = tk.Frame(root, width=300, height=300, bg='pink')
    bottom_right_frame = tk.Frame(root, width=300, height=300)

    top_left_frame.grid(row=0, column=0, sticky="nsew")
    camera_feed_label = tk.Label(top_left_frame)
    camera_feed_label.pack()
    bottom_left_frame.grid(row=1, column=0, sticky="nsew")

    top_right_frame.grid(row=0, column=1, sticky="nsew")
    bottom_right_frame.grid(row=1, column=1, sticky="nsew")

    log_text = tk.Text(bottom_right_frame, wrap="word", bg="black", fg="white")
    log_text.pack(fill="both", expand=True)

    scrollbar = tk.Scrollbar(bottom_right_frame, command=log_text.yview)
    scrollbar.pack(side="right", fill="y")
    log_text.config(yscrollcommand=scrollbar.set)


    update_overlay_feed(top_left_frame)

    update_camera_feed(bottom_left_frame)



    forward_button = tk.Button(
        top_right_frame,
        text="    ↑    ",
        command=lambda: [
                         log_forward()]
    )

    backward_button = tk.Button(
        top_right_frame,
        text="    ↓    ",
        command=lambda: [
                         log_backward()]
    )

    left_button = tk.Button(
        top_right_frame,
        text="    ←    ",
        command=lambda: [
                         log_left()]
    )

    right_button = tk.Button(
        top_right_frame,
        text="    →    ",
        command=lambda: [
                         log_right()]
    )
    play_button = tk.Button(
        top_right_frame,
        text="   ▶   ",
        command=lambda: [
                         log_play()]
    )
    stop_button = tk.Button(
        top_right_frame,
        text="   🟥   ",
        command=lambda: [
                         log_stop()]
    )

    logout_button = tk.Button(top_right_frame, text="   Logout   ", command=lambda: [
                                                                                     log_logout()])

    forward_button.grid(row=1, column=1)
    backward_button.grid(row=3, column=1)
    left_button.grid(row=2, column=0)
    right_button.grid(row=2, column=2)
    stop_button.grid(row=2, column=1)
    play_button.grid(row=4, column = 1)
    logout_button.grid(row=5, column=1)

    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    current_time = datetime.datetime.now()
    today = current_time.strftime("%x")
    today_time = current_time.strftime("%X").replace(':', '꞉')
    filename = f"{username} {today} {today_time}.txt"
    today = today.replace('/', '-')
    script_directory = os.path.dirname(__file__)
    filename = f"{script_directory}\\{username} {today} {today_time}.txt"

    with open(filename, 'w') as f:
        f.write(f'New Log File @{username}!')
        print("log created")

    #the following log_ functions add the movements to the log file
    def log_forward():
        current_time = datetime.datetime.now()

        today = current_time.strftime("%x")
        today_time = current_time.strftime("%X")
        today = today.replace('/', '-')
        log_message(f'\n{username}/Move Forward {today} {today_time}')
        with open(filename, 'at') as f:
            f.write(f'\n{username}/Move Forward {today} {today_time}')

    def log_backward():
        current_time = datetime.datetime.now()

        today = current_time.strftime("%x")
        today_time = current_time.strftime("%X")
        today = today.replace('/', '-')
        log_message(f'\n{username}/Move Backward {today} {today_time}')
        with open(filename, 'at') as f:
            f.write(f'\n{username}/Move Backward {today} {today_time}')

    def log_left():
        current_time = datetime.datetime.now()

        today = current_time.strftime("%x")
        today_time = current_time.strftime("%X")
        today = today.replace('/', '-')
        log_message(f'\n{username}/Turn Left {today} {today_time}')
        with open(filename, 'at') as f:
            f.write(f'\n{username}/Turn Left {today} {today_time}')

    def log_right():
        current_time = datetime.datetime.now()

        today = current_time.strftime("%x")
        today_time = current_time.strftime("%X")
        today = today.replace('/', '-')
        log_message(f'\n{username}/Turn Right {today} {today_time}')
        with open(filename, 'at') as f:
            f.write(f'\n{username}/Turn Right {today} {today_time}')

    def log_play():
        current_time = datetime.datetime.now()

        today = current_time.strftime("%x")
        today_time = current_time.strftime("%X")
        today = today.replace('/', '-')
        log_message(f'\n{username}/Play {today} {today_time}')
        with open(filename, 'at') as f:
            f.write(f'\n{username}/Play {today} {today_time}')
    def log_stop():
        current_time = datetime.datetime.now()

        today = current_time.strftime("%x")
        today_time = current_time.strftime("%X")
        today = today.replace('/', '-')
        log_message(f'\n{username}/Stop {today} {today_time}')
        with open(filename, 'at') as f:
            f.write(f'\n{username}/Stop {today} {today_time}')

    def log_logout():
        #global log_text
        current_time = datetime.datetime.now()

        root.destroy()
        today = current_time.strftime("%x")
        today_time = current_time.strftime("%X")
        today = today.replace('/', '-')
        #log_message(f'\n{username}/Logout {today} {today_time}')
        with open(filename, 'at') as f:
            f.write(f'\n{username}/Logout {today} {today_time}')
        window()
    #Edits real time log in GUI
    def log_message(message):
        current_text = log_text.get("1.0", tk.END)
        new_text = f"{message}\n{current_text}"
        log_text.delete("1.0", tk.END)
        log_text.insert("1.0", new_text)
    #Just to say welcome + name
    def welcome():
        log_text.insert("1.0", f"\nWelcome {name}\n")

    welcome()

    root.mainloop()

# Save window
def save():

    root = Tk()
    root.geometry("600x400")
    root.title("Save and Exit")
    label3 = Label(root, text="")
    label3.pack()
    button = Button(root, text='Save and Exit', command=exit)
    button.pack()

if __name__ == '__main__':
    window()