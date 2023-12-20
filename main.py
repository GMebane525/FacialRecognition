import cv2

# get user values
imagePath = 'Test_Photo.jpg'
cascadePath = 'haarcascade_frontalface_default.xml'
# create the haar cascade
faceCascade = cv2.CascadeClassifier(cascadePath)
# read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect the faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2, # size of the face it is looking at
    minNeighbors=5, # how many symbol of a face do we want to look at (lowering increasing sensitivity)
    minSize=(30,30)
)
print('Found {0} faces!' .format(len(faces))) #fills in the {} with # faces
#draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow('Found found', image) #display image
cv2.waitKey(0) #holds the image so it doesn't go away instantly
