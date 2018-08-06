# DQNMoveOnly Results

## Model Details
This model operates on reduced state and action spaces, processing only the `player_relative` screen feature layer, and outputting a spacial coordinate meant to be used as an argument to `Move_screen("now", ...)`.

## Results

## Training Notes/Caveats
* The Experience Replay buffer does not persist over multiple runs, and is repopulated from scratch each time.
* The target network is updated at the beginning of each new run, even when restoring the online network from a checkpoint.
* Tensorboard summaries are written at the beginning of each episode, for the previous episode. The last episode of a run is not summarized.