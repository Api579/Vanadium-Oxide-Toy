

import numpy as np
import matplotlib.pyplot as plt

# Init Parameters from README
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# each memory cell, VO, as tiny resitor(where resistance is adjustable)
# keeps its latest value

# conductance is inverse of resisitance
# resistance, resists current flow

# pulse, short voltage kicks that nudges conductance up (SET) or down (RESET)



# [2] devices ranges from 10 to 100 uS, init as lowest conductance allowed
G_min = 10e-6

# 10 * window = 100, times bigger the highest conductance is than lowest
# "range" for storing values
window = 10

G_max = G_min * window


# Voltage, small voltage applied to test cell, w/o changing much
V_read = 0.1


#alpha is the step size, how big a nudge is from each pulse
# e.g. alpha =0.07 means pulse moves 7% towards SET or RESET
# SET: ceiling, RESET: floor
# percentage of distance remaining, steps gets smaller as limit approaches
# less distance to cover

# 0.07 comes from paper[1], where calue used to train 40 pulses
# from [1] 40 small steps to cover 95%, 
# device 2, moves in small steps, for fine-tuning
# alpha = 1-0.05^(1/40), 40 seps to 0.95, 1.00-0.95 = 0.05
# eqn general name is geometric decay
# follows [1], uses 95% as well, should be ok if its close to 100 but not 100
# if its 100, alpha = 1, then instant to ceiling not gradual
D2_alpha_set = 0.07
D2_alpha_reset = 0.07

#D1 has gradual steps for RESET, but big steps for SET
# its to test if device can accurately store values if only big steps are allowed
D1_alpha_reset = 0.07
D1_alpha_set = [0.3, 0.6, 0.9]

pulses_applied = 40

tolerance = 0.01

max_pulses = 100

# helper fucntions
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#set pulse, add the stepped pulse to device
def set_pulse(G, alpha_set, G_max):
    return G + alpha_set * (G_max - G)

def reset_pulse(G, alpha_reset, G_min):
    return G - alpha_reset * (G - G_min)

# since V=IR, R inverse of G, reads current, read returns current in Amperes
def read_device(G, V_read):
    return G * V_read

def verification(G, target, SET, RESET, G_min, G_max, tolerance, max_pulses):
    pulses = 0
    while (abs(G-target)>(tokerance*target))and(pulses<max_pulses):
        if G < target:
            G=set_oulse(G, SET, G_max)
        else:
            G=RESET(G,RESEY, G_min)
        pulses += 1
        error = (G-target)/target
        output = [G,pulses,errot]
        return output

# first test
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def pulse_test_1(alpha_set, alpha_reset, G_min, G_max, num_of_pulses):
    
    G = G_min
    history = [G]
    
    for i in range(num_of_pulses):
        G = set_pulse(G, alpha_set, G_max)
        history.append(G)
        
    for i in range (num_of_pulses):
        G = reset_pulse(G, alpha_reset, G_min)
        history.append(G)
        
    return history


# stores D2 test as 1D array
D2_test = pulse_test_1(D2_alpha_set, D2_alpha_reset, G_min, G_max, pulses_applied)

# save as a graph
index = range(len(D2_test))
G_in_micro = np.array(D2_test)/1e-6
plt.plot(index, G_in_micro, label="D2 Test")
plt.xlabel("Pulses No.")
plt.ylabel("G (µS)")
plt.legend()
plt.savefig("D2_pulse_curves.png", dpi=150)
plt.show()

for i in range(len(D1_alpha_set)):

    D1_test = pulse_test_1(D1_alpha_set[i], D1_alpha_reset, G_min, G_max, pulses_applied)

    # save as a graph
    index = range(len(D1_test))
    G_in_micro = np.array(D1_test)/1e-6
    plt.plot(index, G_in_micro, label=f"D1 Test {i}")
    plt.xlabel("Pulses No.")
    plt.ylabel("G (µS)")
    plt.legend()
    plt.savefig(f"D1_{i}_pulse_curves.png", dpi=150)
    plt.show()

# Second test
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    
