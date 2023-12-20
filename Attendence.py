import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

anand_image = face_recognition.load_image_file("anand.jpg")
anand_encoding = face_recognition.face_encodings(anand_image)[0]

gil_image = face_recognition.load_image_file("gil.jpg")
gil_encoding = face_recognition.face_encodings(gil_image)[0]

known_face_encoding = [
    anand_encoding,
    gil_encoding
]

known_faces_names = [
    "anand",
    "gil"
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()  # get date
current_date = now.strftime("%Y-%m-%d")  # only want the date

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)  # write to csv

while True:
    _, frame = video_capture.read()  # get data from webcam
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # make smaller
    rgb_small_frame = small_frame[:, :, ::-1]  # convert for bgr to rgb
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)  # is there a face in the frame?
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  # stores the face data from the frame
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)  # compairing known faces to the data from the webcam
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)  # are these faces likely to be the same?
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]  # name of the face that was recognized

            face_names.append(name)  # enter name in csv file
            if name in known_faces_names:
                if name in students:
                    students.remove(name)  # don't want multiple outputs
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])
    cv2.imshow("attendance system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

