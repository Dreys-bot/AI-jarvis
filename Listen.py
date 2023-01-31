import speech_recognition as sr


def Listen():
    r = sr.Recognizer() #create a instance of the class speechrecognition

    with sr.Microphone() as source:
        print('Listening...')
        r.pause_threshold=1
        audio = r.listen(source, 0, 3)

    try:
        print("recognizing...")
        query= r.recognize_google(audio, language="en-in")
        print(f"You said: {query}")
    except Exception as e:
        print(e)
        print("Unable to Recognize your voice.")
        return "None"

    return query


