import pygame

pygame.init()
alarm_sound = pygame.mixer.Sound('alarm.wav')

def start_alarm():
    print("Starting alarm!")  
    alarm_sound.play(maxtime=5000)  

def stop_alarm():
    print("Stopping alarm!") 
    alarm_sound.stop()
