# DQNMoveOnly Results

## Model Details

### Inputs/Outputs
This model operates on reduced state and action spaces, processing only the `player_relative` screen feature layer, and outputting a spacial coordinate meant to be used as an argument to `Move_screen("now", ...)`.

### Estimator
A deep Q network is used to model the value of each action conditioned on the state. The value is equal to the expected reward due to selecting the action, plus the value of the best action possible from the state the environment will transition to as a result of the action - discounted multiplicatively by `discount_factor`.

### Action Selection
A epsilon-greedy strategy is used, where the probability of selecting a random action, epsilon, is annealed linearly from a maximum value `epsilon_max` to a minimum value `epsilon_min` over a number of steps equaling `epsilon_decay_steps`. When selecting an action using the model, the action having the maximum estimated value (Q) according to the online DQN is chosen.

## Results


## Training Notes/Caveats
* The Experience Replay buffer does not persist over multiple runs, and is repopulated from scratch each time.
* The target network is updated at the beginning of each new run, even when restoring the online network from a checkpoint.
* Tensorboard summaries are written at the beginning of each episode, for the previous episode. The last episode of a run is not summarized.