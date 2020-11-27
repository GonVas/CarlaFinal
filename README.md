# CarlaFinal

![View of sensors](/diagrams/bl_integrated_out.png)

Master Thesis about multi-agent Deep Reinforcement Learning for autonomous driving.

First Carla was transformed into an openAI Gym enverioment and then several architecture for both an encoder and a policy network were used.

First a VAE was used as an encoder but it wasn't powerfull enough to encode the input images and when the policy netwokr "looked" at the latent space it couldn't barely keep a vehicle in lanes.

Next an encoder using an RESNET was used an it performed way better, on the policy network side, it was a LSTM trained with SAC (altough other algos are implemented SAC performed better).

## Final Algorithm - SACMessage

The main contribution of this work is a new algortihm composed SAC (Soft-Actor-Critic), A3C (making it distributed) and Geometric Deep Learning, by making the different agents (vehicles) a graph a message parssing algorithm is used.



General Diagram:

![General Diagram](/diagrams/carla_diagram_paper.png)

Encoder:
In order for the RL agents to perform better, the encoder needs human features which are added in the `CarlaGymEnv.py` before the step function that gives the input to the agent. 
TODO: Change to separate file, like encoder.py

![Encoder Diagram](/diagrams/lanedete.png)

RL Diagram:
The algoritm uses as a base SAC (Soft-Actor-Critic).
![RL Diagram](/diagrams/graph_message.png)
