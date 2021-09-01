import streamlit as st 
import os
import numpy as np
import cv2
from keras.models import load_model

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))

st.header("Hello world !")
st.balloons()

def local_css(file_name):
    # with open(file_name) as f:
    #     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    os.open(file_name, os.O_RDWR)
def about():
	st.write(
		'''
        Face Features Detection by Machine Learning Knowledge

        This app is built by using OpenCV library and Streamlit package.
		''')

def main():
    """Face Features Detection App"""

    local_css('style.css')

    st.image("AmogUs.png")

    t = "<h2 class='title blue'>Face Features Detection App</h2>"

    st.markdown(t,unsafe_allow_html=True)

    te = "<div class='title blue'>Built with OpenCV and Streamlit</div>"

    st.markdown(te,unsafe_allow_html=True)

    # displaying emotion on webpage
    # instructions for how to use
    # having a button for user to start recording
    # different file for recording
    # push to heroku

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)

    if choice == "Home":
        st.write('Press ')
        emotion_model = load_model('model.h5')
        emotion_model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

        # print('hello')

        cv2.ocl.setUseOpenCL(False)

        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        cap = cv2.VideoCapture(0)
        results = [0, 0, 0, 0, 0, 0, 0]
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            bounding_box = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            num_faces = bounding_box.detectMultiScale(image = gray_frame, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                results[maxindex] += 1
                cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                st.write()

            cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

        # analyze "results" and give suggestions accordingly
        total_frames = 0
        for i in range(len(results)):
            total_frames += results[i]

        positive_frames = results[3] + results[4] + results[6]
        positive_percentage = int(positive_frames / total_frames * 100)
        st.write('You were positive for ' + str(positive_percentage) + 'percent of the time.')
        if (positive_percentage < 20):
            st.write('You seem to be in high stress. Consider taking a walk outside and a few deep breaths.')

        

    elif choice == "About":
    	about()




if __name__ == "__main__":
    main()