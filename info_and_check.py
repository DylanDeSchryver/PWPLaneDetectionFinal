from tkinter import messagebox


from sql_program import save_sql, get_sql, first


# Save user information to global variables and run save_sql()
def save_info(creates,user_info,name_entry, last_entry, user_entry, passw_entry):

    user_info['name'] = name_entry.get()
    user_info['last_name'] = last_entry.get()
    user_info['username'] = user_entry.get()
    user_info['password'] = passw_entry.get()
    save_sql()

    creates.destroy()


# Check if the username and password are correct
def check_login(logins,userlogin_entry, passwlogin_entry):


    username = userlogin_entry.get()
    password = passwlogin_entry.get()
    key = get_sql(username, password)

    if key:
        from GUI import loggedin
        logins.destroy()
        name = first(username, password)
        
        loggedin(username, password, key, name)
    else:
        messagebox.showerror('Error', "Incorrect Username or Password.")
        logins.destroy()
