import sqlite3
import random
import GUI
import hashlib
import base64


def save_sql():
    user_info = GUI.user_info  # gets dictionary from GUI, which should be filled in from create or login functions

    connection = sqlite3.connect("USER.db")  # makes connection to table and database

    crsr = connection.cursor()
    # creates table
    sql_command = """CREATE TABLE IF NOT EXISTS user ( 
  Key INTEGER PRIMARY KEY, 
  fname VARCHAR(20), 
  lname VARCHAR(30), 
  user VARCHAR(30), 
  pass VARCHAR(30));"""

    crsr.execute(sql_command)

    key = random.randrange(100000, 999999)

    crsr.execute(
        "SELECT key FROM user WHERE key = {key}".format(key=key))  # generate key, make sure it does not exist already
    ans = crsr.fetchall()
    while ans != []:
        key = random.randrange(100000, 999999)

    crsr.execute(
        'INSERT INTO user VALUES ({key}, "{name}", "{last}", "{user}", "{passw}")'.  # insert values
        format(key=key, name=user_info['name'], last=user_info['last_name'], user=user_info['username'],
               passw=hashlib.sha256(user_info[
                                        'password'].encode()).hexdigest()))  # encrypt password using hashlib sha256 and base64 libraries

    crsr.execute("SELECT * FROM user")

    ans = crsr.fetchall()

    for x in ans:
        print(x)

    connection.commit()

    connection.close()


def get_sql(username, password):
    connection = sqlite3.connect("USER.db")  # connect to database and table
    crsr = connection.cursor()

    crsr.execute("SELECT key FROM user WHERE user = '{user}' AND pass = '{passw}'".format(user=username,
                                                                                          passw=hashlib.sha256(
                                                                                              password.encode()).hexdigest()))

    ans = crsr.fetchall()  # ans is the key for the account in which the username and password are equal to the ones entered in the login.

    return ans  # this returns ans back to the original login...there we can login using this key
    connection.close()


def first(username, password):
    connection = sqlite3.connect("USER.db")  # connect to database and table
    crsr = connection.cursor()

    crsr.execute("SELECT fname FROM user WHERE user = '{user}' AND pass = '{passw}'".format(user=username,
                                                                                            passw=hashlib.sha256(
                                                                                                password.encode()).hexdigest()))

    ans = crsr.fetchall()  # ans is now first name

    ansl = list(ans)

    def tuptostr(tup):
        global ansl
        # initialize an empty string
        empty = ''
        for x in tup:
            empty = empty + x
        return empty

    tuple = ansl[0]
    ansl = tuptostr(tuple)

    return ansl
    connection.close()