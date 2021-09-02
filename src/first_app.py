from webbrowser import Chrome
import streamlit as st 
import time
import os
import numpy as np
import cv2
from keras.models import load_model
from bokeh.plotting import figure, show, output_file

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def about():
	st.write(
		'''
        Emotion Detection and Well-Being Monitor with the power of AI.

        This app is built using OpenCV, Streamlit package, Tensorflow/Keras, the Emojify project (Data Flair), and the generous help of the incredible team of SPIS mentors.

        We want to finish here by saying our sincerest thanks to the tremendous team behind SPIS. We have both had a blast and appreciate all the hard work everyone had put in.

        Gracias mi amigos.

        No, no, Tim. No me hablo español.

        Thank you. :^) 
                                    
                                --Steven and Tim
		'''
    )

def instructions():
    st.write(
        '''
        This is an app designed to run in the background while you work.
        It will monitor your emotions and give you suggestions to improve your mental health.
        
        To start, simply navigate to the Video Capture tab, and the app will begin recording shortly.
        Press q to quit the app at any time, and you will see your results on the Video Capture page.
        '''
    )

def main():
    """Emotion Detector and Well-Being Monitor"""

    local_css('src/style.css')


    activities = ["Home", "About", "Video Capture"]
    choice = st.sidebar.selectbox("Pick something fun", activities)

    if choice == "Video Capture":
        st.header("Video Capture:")
        emotion_model = load_model('src/model.h5')
        emotion_model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

        cv2.ocl.setUseOpenCL(False)

        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
  
  
        # input time in seconds
        timect = 60 * st.number_input("Enter the time in minutes: ", min_value=1, max_value=120)
        start_time = time.time()
        end_time = start_time + timect
        
        cap = cv2.VideoCapture(0)
        results = [0, 0, 0, 0, 0, 0, 0]
        frames = 0
        emotion_score_list = [0]
        time_elapsed_list = [0]
        emotion_score = 0

        t = st.empty()
        temptime = 0
        while time.time() < end_time:
           
            mins, secs = divmod((int(end_time - time.time())), 60)
            timer = '{:02d}:{:02d}'.format(mins, secs)
            if(timer == temptime):
                t.empty()
            t.write('Time Remaining: ' + str(timer))
            temptime = timer
            
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
                # tentative code
                if (emotion_dict[maxindex] == 'Happy' or emotion_dict[maxindex] == 'Surprised'):
                    emotion_score += 1
                elif (emotion_dict[maxindex] == 'Neutral'):
                    pass
                else:
                    emotion_score -= 1

                frames += 1
                
                if (frames % 60 == 0):
                    emotion_score_list.append(emotion_score)

            cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

        # analyze "results" and give suggestions accordingly
        total_frames = 0
        for i in range(len(results)):
            total_frames += results[i]

        positive_frames = results[3] + results[6]
        positive_percentage = int(positive_frames / total_frames * 100)
        st.write('You were positive for ' + str(positive_percentage) + ' percent of the time.')
        if (positive_percentage < 50):
            random_number = np.random.randint(0, 2)
            if (random_number == 1):
                st.write('You seem to be in high stress. Consider taking a walk outside, watching some memes on YouTube, or resorting to a few deep breaths.')
            else:
                st.write('Consider purchasing a stress ball or playing Kick the Buddy. Its up to you to squeeze or punch, but it is probably better than what you are doing at the moment.')
        elif (positive_percentage > 75):
            random_number = np.random.randint(0,2)
            if (random_number == 1):
                st.write('You seem suspiciously happy in this world of constant pressure. Keep it up. Lets see how long you could last...')
            else:
                st.write('Are you actually doing your work? You seem too happy to me.')
        else:
            random_number = np.random.randint(0,2)
            if (random_number == 1):
                st.write('Good job creating the mean for all of us to regress to. Keep it up!')
            else:
                st.write('Everyone is stressed at some point in time. Take it easy. Stay normal. :|')


        for i in range(len(emotion_score_list) - 1):
            time_elapsed_list.append(i + 1)

        output_file('stress_model_graph.html')
        p = figure(
            title='Your Emotional Well-Being Over Time',
            x_axis_label='Time Elapsed (60 frames)',
            y_axis_label='Emotional Score')

        p.line(time_elapsed_list, emotion_score_list, legend_label='Trend', line_width=2)
        st.bokeh_chart(p, use_container_width=True)
        show(p)
       
    elif choice == "About":
        st.header("About:")
        about()
        st.balloons()

    elif choice == "Home":
        t = "<h2 class='title blue'>Face Features Detection App</h2>"

        st.markdown(t,unsafe_allow_html=True)

        te = "<div class='title blue'>Built with OpenCV and Streamlit</div>"

        st.markdown(te,unsafe_allow_html=True)

        st.image("src/AmogUs.png")
        
        instructions()



if __name__ == "__main__":
    main()