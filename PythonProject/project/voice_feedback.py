import pyttsx3, threading, time
engine = pyttsx3.init()
engine.setProperty('rate', 165)
_last = 0

def speak(text):
    global _last
    if time.time() - _last < 2:
        return
    _last = time.time()

    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()
