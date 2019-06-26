import dxinput
import ss
import cnn
import cv2
import time
import neat
import math
import pickle
import numpy as np

pressed = [0 for i in range(5)]
screen_name = "Need for Speed™ Most Wanted"
ini = ss.get_screen(screen_name)

def restart_race():
    time.sleep(1)
    dxinput.PressKey(0x1C)
    time.sleep(0.2)
    dxinput.ReleaseKey(0x1C)
    time.sleep(1.5)
    dxinput.PressKey(0x02)
    time.sleep(0.2)
    dxinput.ReleaseKey(0x02)
    time.sleep(1.5)
    dxinput.PressKey(0xCB)
    time.sleep(0.2)
    dxinput.ReleaseKey(0xCB)
    time.sleep(1.5)
    dxinput.PressKey(0x1C)
    time.sleep(0.2)
    dxinput.ReleaseKey(0x1C)
    time.sleep(3.1)

def take_action(result):
    for i in range(5):
        if i == 0:
            key = 0xC8
        elif i == 1:
            key = 0xD0
        elif i == 2:
            key = 0xCB
        elif i == 3:
            key = 0xCD
        else:
            key = 0x38
        if result[i] >= 0.5:
            if pressed[i] == 0:
                pressed[i] = 1
                dxinput.PressKey(key)
        else:
            if pressed[i] == 1:
                pressed[i] = 0
                dxinput.ReleaseKey(key)

def clear_presses():
    for i in range(5):
        if i == 0:
            key = 0xC8
        elif i == 1:
            key = 0xD0
        elif i == 2:
            key = 0xCB
        elif i == 3:
            key = 0xCD
        else:
            key = 0x38
        if pressed[i] == 1:
            dxinput.ReleaseKey(key)
            pressed[i] = 0


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.RecurrentNetwork.create(genome,config)
        score = 0
        speed = 0
        restart_race()
        start = time.process_time()
        last_lap = start
        lap = start

        while True:
            img = ss.get_screen(screen_name)
            img,speed = cnn.get_speed(img)
            img = np.reshape(img, (660))
            result = net.activate(img)
            take_action(result)
            lap = time.process_time()
            score += (lap-last_lap)*speed
            if lap - start > 150:
                print("Time up!")
                break
            if score < math.floor((lap-start)/20)*1500:
                print("Not making progress")
                break
            # print(score)
            last_lap = lap
        
        clear_presses()
        genome.fitness = score[0]
        print("Genome: ", genome_id,"\n Score: ", score[0])


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        "config.txt")
p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

winner = p.run(eval_genomes, 100)

print('\nBest genome:\n{!s}'.format(winner))

with open("winner.pkl", 'wb') as output:
    pickle.dump(winner,output,1)