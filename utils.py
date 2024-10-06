import math
import numpy as np
import random

np.random.seed(99)
random.seed(99)


def quantitySkew(numberSample, numberClient):
    minSample = math.floor(0.3 / numberClient * numberSample)
    remaining_elements = numberSample - numberClient * minSample
    group_sizes = [minSample] * numberClient

    # Define the integers and their corresponding probabilities
    integers = [i for i in range(numberClient)]
    # gererate random probabilities for each integer
    probabilities = np.random.dirichlet(np.ones(numberClient), size=1)[0]
    for _ in range(remaining_elements):
        group_sizes[np.random.choice(integers, p=probabilities)] += 1
    return group_sizes


def generate_random_color():
    """
    Generate a random color in hexadecimal format.
    """
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)
