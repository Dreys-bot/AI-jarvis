#Function
import datetime
from speak import speak
#2 Types

#1 - Non Input
#eg: Time, Date, Speedtest

def Time():
    time = datetime.datetime.now().strftime("%H:%M")
    speak(time)

def Date():
    date = datetime.date.today()
    speak(date)

def Day():
    day = datetime.datetime.now().strftime("%A")
    speak(day)


def NonInputExecution(query):

    query = str(query)

    if "time" in query:
        Time()
    elif "date" in query:
        Date()
    elif "day" in query:
        Day()

#2 - Input
#eg - google search, wikipedia



def InputExecution(tag, query):
    if "wikipedia" in tag:
        name = str(query).replace("who is", "").replace("about", "").replace("what is", "").replace("wikipedia", "")
        import wikipedia
        result = wikipedia.summary(name)
        speak(result)
    elif "google" in tag:
        query = str(query).replace("google", "")
        query = query.replace("search", "")
        import pywhatkit
        pywhatkit.search(query)

#Open file in computer