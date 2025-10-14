# evaluator.py
import math, random
import neat

# 2‑input XOR data
xor_inputs  = [(0.0,0.0),(0.0,1.0),(1.0,0.0),(1.0,1.0)]
xor_outputs = [(1.0,0.0),(0.0,1.0),(0.0,1.0),(1.0,0.0)]
num_evaluations = 25




def eval_genome(genome, config):
    """Evaluate a single genome; return its fitness."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    max_error = 0.0
    for _ in range(num_evaluations):
        i = random.randrange(len(xor_inputs))
        xi, xo = xor_inputs[i], xor_outputs[i]
        out = net.activate(xi)
        err = math.sqrt((out[0]-xo[0])**2 + (out[1]-xo[1])**2)
        max_error = max(max_error, err)
    return 1.0 - max_error
