import threading

import pyautogui as pg


def check_for_quit():
    import keyboard  # This import must be inside the thread function
    while True:
        if keyboard.is_pressed('q'):
            print("Stopping the program as 'q' is pressed")
            break

# Start the thread to check for the 'q' key press
quit_thread = threading.Thread(target=check_for_quit)
quit_thread.start()

# Main loop to print the mouse position
while quit_thread.is_alive():
    print(pg.position())
