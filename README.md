# Abstract

Brain simulations are often used in neuroscience to test theoretical models of computation. Understanding the low-level dynamics of the brain is fundamental to develop better artificial intelligence algorithms, and in turn those algorithms provide more sophisticated methods to crack the brain code. 
Still, building simulations of sections of the brain is complex, and finding the parameters of the neurons and synapses so that the behavior of the artificial network matches the biological one requires a lot of time and resources.
Here it is developed a simple framework to automate the search for the best parameters for the simulation, and is provided a use case with a model of a hypercolumn and a Reinforcement Learning algorithm to optimize the parameters so that the firing rate of the excitatory populations in every layer matches the activity recorded in vivo.
The results show that Reinforcement Learning obtains results that exceed in accuracy the referenced papers.
ing on it after the winter session.

# Structure

- The agent directory contains the code for Deep Deterministic Policy Gradient algorithm, adapted for a multi-armed bandit setup
- The env directory contains the code for the brain simulation, supplied by NEST in their source code, developed following the Potjans and Diesmann (2014) work.
