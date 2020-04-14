import random
import pygame
import math

# Instead of using a constant in the while loop
minMovement = 0.5
maxSpeed = 5

# Create class person
class Person():

    status_colors = {'susceptible':'white', 'infected':'red', 'recovered':'blue'}

    def __init__(self, x, y, status, socialDistancing):
        """
        Initialize person

        Position in grid: x, y. All x and y are unique and random at start.
        So no spawning on top of each other. Thats why they are entered now.
        Status: susceptible, infected, recovered.
        Status is in here because patient zero must be assigned with start value 1.
        SocialDistancing: 0 velocity (not moving on the screen)
        """
        # Define each variable
        self.x = x # will become floats once adding velocity
        self.y = y
        self.status = status
        self.socialDistancing = socialDistancing
        # An individual
        self.radius = 6
        # Velocity in x and y direction. Value = 0 since assume socialDistancing
        self.vx = self.vy = 0
        # Count how many turns this person is sick
        # Only patient zero starts with value 1
        self.turnSick = 0
        # If put here, then all have random recoveryTimes
        self.recoveryTime = random.randint(100,150) # frames
        # Give them a random (uniform) velocity for their movement if not socialDistancing
        if not self.socialDistancing:
            # Add a minimum speed to avoid very small values. Keep picking new random values.
            while -minMovement < self.vx < minMovement and -minMovement < self.vy < minMovement:
                self.vx = random.uniform(-maxSpeed,maxSpeed)
                self.vy = random.uniform(-maxSpeed,maxSpeed)

    def draw(self, screen):
        """
        Draws the person (circle with radius) on x,y in the PyGame screen (main surface).
        """
        # pygame requires integers, therefore, round
        # color: set of colors using dictionary. self.colors to access dictionary, and inside you call the status
        pygame.draw.circle(screen, pygame.Color(self.status_colors[self.status]), (round(self.x), round(self.y)), self.radius)

    def move(self):
        """
        Updates x and y based on the new velocities
        """
        # Little check to make sure. Avoids potential errors.
        if not self.socialDistancing:
            self.x += self.vx
            self.y += self.vy

    def checkCollidingWithWall(self, screen):
        """
        Prevents the person from moving off the screen.
        """
        # self.vx > 0 makes sure there is actually a velocity, and prevent getting stuck to the wall
        # For both x sides of the screen
        if self.x + self.radius >= screen.get_width() and self.vx > 0:
            # Switch velocity direction
            self.vx *= -1
        elif self.x - self.radius <= 0 and self.vx < 0:
            self.vx *= -1
        if self.y + self.radius >= screen.get_height() and self.vy > 0:
            self.vy *= -1
        elif self.y - self.radius <= 0 and self.vy < 0:
            self.vy *= -1

    def checkCollidingWithOther(self, other):
        """
        Checks if colliding with other (person)

        Used once to prevent spawning on top of each other
        Used during simulation to change the velocities
        """
        # Pythagoras
        distance = math.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)
        if distance <= (self.radius + other.radius):
            return True # then the objects are colliding
        else: # not necessary actually
            return False

    def updateCollisionVelocities(self,other):
        """
        Change velocities when colliding

        Type 1 collision: both objects are moving, so neither is SocialDistancing
        Switch velocities.

        Type 2 collision: one object that is socialDistancing (not moving) and
        one that is not socialDistancing (moving). We don't want the person that
        is socialDistancing to start moving. Just change the velocity of the person
        that is socialDistancing.
        """
        # Type 1:
        if not self.socialDistancing and not other.socialDistancing:
            # Use temporary to prevent the value from getting lost.
            tempVX = self.vx
            tempVY = self.vy
            self.vx = other.vx
            self.vy = other.vy
            other.vx = tempVX
            other.vy = tempVY
        # Type 2:
        elif other.socialDistancing:
            # # Works okay, but not great
            # self.vx *= -1
            # self.vy *= -1
            # # Works okay
            # tempVX = self.vx
            # self.vx = self.vy
            # self.vy = tempVX
            # Vector math approach
            # Magnitude of velocity vector (= length of vector)
            magV = math.sqrt((self.vx)**2 + (self.vy)**2)
            # Describes the new direction (vector) the old vector should go
            tempVector = (self.vx + (self.x - other.x), self.vy + (self.y - other.y))
            # Magnitude of tempVector
            magTempVector = math.sqrt((tempVector[0])**2 + (tempVector[1])**2)
            # Normalize the new vector (pointing in the direction it should go in)
            normTempVector = (tempVector[0]/magTempVector, tempVector[1]/magTempVector)
            # Apply original Velocity to the new vector
            self.vx = normTempVector[0] * magV
            self.vy = normTempVector[1] * magV

    def update(self, screen, people):
        """
        Execute/update once per frame.

        Moves, checks for collisions.
        """
        self.move()
        # Check to see how many turns they were sick, and if their turnSick == recoveryTime, set status to recovered
        # They must already be sick
        if self.status == 'infected':
            self.turnSick += 1
            # Putting it in the other if saves an additional check.
            if self.turnSick == self.recoveryTime:
                self.status = 'recovered'
        # Check for collisions with wall
        self.checkCollidingWithWall(screen)
        # Check for collision with other people
        for other in people:
            # Check if not colliding with itself
            if self != other:
                # If collision...
                if self.checkCollidingWithOther(other):
                    # Update collision velocities
                    self.updateCollisionVelocities(other)
                    # The other infects you, or you infect the other.
                    if self.status == 'infected' and other.status == 'susceptible':
                        other.status = 'infected'
                    elif other.status == 'infected' and self.status == 'susceptible':
                        self.status = 'infected'
