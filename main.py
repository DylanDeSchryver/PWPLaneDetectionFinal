import cv2
from GUI import window
import os

file_path = "USER.db"

if not os.path.exists(file_path):
    open(file_path, 'w').close()



window() # Line detection

