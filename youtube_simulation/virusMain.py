import pygame
import random

import person
from person import Person

"""
What else to do:

- Show plots/statistics over time
- Make variables adjustable (dashboard possibilities?)
- Remove the velocity that changes once colliding (is this representative of real life?)

"""

def main():
    # Initialize pygame
    pygame.init()
    # Create pygame window
    WIDTH = HEIGHT = 600
    # Used to draw in later
    screen = pygame.display.set_mode([WIDTH, HEIGHT])
    pygame.display.set_caption('Corona simulation')
    screen.fill(pygame.Color('gray'))
    # Control the framerates with the clock function
    clock = pygame.time.Clock()
    MAX_FPS = 20

    # For our running simulation program
    running = True
    spawnBuffer = 10 # pixels
    numPeople = 200
    factorSocialDistancing = 0.75

    # Create people
    # Place patient zero at a random position in the screen (not practicing socialDistancing)
    patientZero = Person(random.randint(spawnBuffer,WIDTH-spawnBuffer), random.randint(spawnBuffer,HEIGHT-spawnBuffer), 'infected', False)
    # Other people
    people = [patientZero]
    for i in range(numPeople-1):
        socialDistancing = False
        if i < factorSocialDistancing * numPeople: # Those who are doing socialDistancing
            socialDistancing = True

        # Make sure that each spawn location is unique
        colliding = True
        while colliding:
            person = Person(random.randint(spawnBuffer,WIDTH-spawnBuffer), random.randint(spawnBuffer,HEIGHT-spawnBuffer), 'susceptible', socialDistancing)
            colliding = False
            for p in people:
                if person.checkCollidingWithOther(p):
                    colliding = True
                    break
        people.append(person)

    while running: # While the app is running
        # to empty the event queue
        for event in pygame.event.get():
            # Check if user clicked top right X to close program
            if event.type == pygame.QUIT:
                running = False
        # Update people
        # # Update patient zero
        # patientZero.update(screen, [])
        for person in people:
            person.update(screen, people)
        # Update graphics
        # If you don't do this, you will see a trace after the person.
        screen.fill(pygame.Color('gray'))
        # # Draw patient zero
        # patientZero.draw(screen)
        # Draw all people
        for person in people:
            person.draw(screen)
        # Update - crucial
        pygame.display.flip()
        # Pauze for frame to be displayed
        clock.tick(MAX_FPS)
    # If quit, stop the program
    pygame.quit()

# Call the main class here
main()
