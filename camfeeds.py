import tkinter as tk
from tkinter import *
from PIL import ImageTk
from PIL import Image
import cv2
from lineDetect import lineDetection

# Function to update camera feed with overlay
def update_overlay_feed(top_left_frame):
    try:
        url = 'video.mp4'
        cap = cv2.VideoCapture(url)

        # Function to convert OpenCV image to Tkinter PhotoImage
        def convert_to_photo_image(frame):
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image=image)
            return photo

        # Function to update the label within top_left_frame with a new image
        def update_label():
            ret, frame = cap.read()

            if ret:
                frame = cv2.resize(frame,(650,380))
                photo = convert_to_photo_image(frame)
                camera_feed_label.configure(image=photo)
                camera_feed_label.image = photo
                top_left_frame.after(10, update_label)  # Update every 10 milliseconds
            else:
                print("Error reading frame from camera feed")

        # Create a label within top_left_frame for displaying the camera feed
        camera_feed_label = tk.Label(top_left_frame)
        camera_feed_label.pack()

        # Start updating the label
        update_label()

    except Exception as e:
        print(f"Error updating camera feed: {e}")

#Function to display camera feed without overlay
def update_camera_feed(bottom_left_frame):
    try:
        url = 'videoc.mp4'
        cap = cv2.VideoCapture(url)

        # Function to convert OpenCV image to Tkinter PhotoImage
        def convert_to_photo_image(frame):
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image=image)
            return photo

        # Function to update the label within top_left_frame with a new image
        def update_label():
            ret, frame = cap.read()

            if ret:
                frame = lineDetection(cap)
                frame = cv2.resize(frame, (650, 380))
                photo = convert_to_photo_image(frame)
                camera_feed_label.configure(image=photo)
                camera_feed_label.image = photo
                bottom_left_frame.after(10, update_label)  # Update every 10 milliseconds
            else:
                print("Error reading frame from camera feed")

        # Create a label within top_left_frame for displaying the camera feed
        camera_feed_label = tk.Label(bottom_left_frame)
        camera_feed_label.pack()

        # Start updating the label
        update_label()

    except Exception as e:
        print(f"Error updating camera feed: {e}")