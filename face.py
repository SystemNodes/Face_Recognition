import cv2
import streamlit as st

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier('C:\\Users\\patri\\Documents\\Face Recognition\\haarcascade_frontalface_default.xml')


def hex_to_rgb(hex_color):
    """Converts a hexadecimal color string to an RGB tuple."""
    # Remove '#' from the hexadecimal color string
    hex_color = hex_color.lstrip('#')
    # Convert the hexadecimal color string to RGB tuple
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return rgb_color


def detect_faces(min_neighbors, scale_factor, rectangle_color):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()
        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)
            # Save the images with detected faces on the user's device
            cv2.imwrite('detected_face.jpg', frame)
        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Welcome to the Face Detection System!")
    st.write("Instructions: Adjust the parameters below to customize face detection, then click on 'Detect Faces' to start.")
    
    # Add a feature to adjust the minNeighbors parameter
    min_neighbors = st.slider("minNeighbors", 1, 10, 5)
    
    # Add a feature to adjust the scaleFactor parameter
    scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.3)
    
    # Add a feature to choose the color of the rectangles drawn around the detected faces
    rectangle_color_hex = st.color_picker("Choose color for rectangles", "#00ff00")
    rectangle_color = hex_to_rgb(rectangle_color_hex)
    
    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        # Call the detect_faces function with the chosen parameters
        detect_faces(min_neighbors, scale_factor, rectangle_color)


if __name__ == "__main__":
    app()
