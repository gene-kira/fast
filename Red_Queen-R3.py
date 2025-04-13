import os
from gtts import gTTS
import playsound
import subprocess
import psutil

# Function to convert text to speech and play it
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    playsound.playsound("output.mp3")
    os.remove("output.mp3")

# Function to check for running processes and terminate them
def terminate_process(process_name):
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == process_name:
            try:
                p = psutil.Process(proc.info['pid'])
                p.terminate()
                speak(f"Terminated process {process_name}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

# Function to simulate a security lockdown
def initiate_lockdown():
    speak("Initiating system lockdown. All non-essential processes will be terminated.")
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if not any([proc.info['name'] == name for name in ['python', 'cmd', 'explorer']]):
                p = psutil.Process(proc.info['pid'])
                p.terminate()
                speak(f"Terminated process {proc.info['name']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

# Function to scan and remove viruses
def scan_and_remove_viruses():
    speak("Initiating virus scan.")
    result = subprocess.run(['malwarebytes', '--scan'], capture_output=True, text=True)
    if "Threats found:" in result.stdout:
        speak(f"Found {result.stdout.split('Threats found:')[1].split('\n')[0]} threats. Initiating removal process.")
        remove_result = subprocess.run(['malwarebytes', '--remove'], capture_output=True, text=True)
        if "Threats removed:" in remove_result.stdout:
            speak(f"Removed {remove_result.stdout.split('Threats removed:')[1].split('\n')[0]} threats.")
    else:
        speak("No viruses detected.")

# Function to handle user commands
def handle_command(command):
    if command == 'lockdown':
        initiate_lockdown()
    elif command.startswith('terminate'):
        process_name = command.split(' ')[1]
        terminate_process(process_name)
    elif command == 'scan':
        scan_and_remove_viruses()
    else:
        speak("Unknown command. Valid commands are: lockdown, terminate [process], and scan.")

# Main loop to interact with the Red Queen
def main():
    speak("Red Queen online. Awaiting commands.")
    while True:
        user_input = input("Enter your command: ").strip().lower()
        handle_command(user_input)

if __name__ == "__main__":
    main()
