# StarCraft II Agents

StarCraft II playing agents that interface directly with Deepmind's [PySC2](https://github.com/deepmind/pysc2) API.

At the moment, deep RL agents are unable to defeat the easiest of the scripted bots in the full game. Therefore, I begin by implementing agents intended to tackle the mini-games introduced and described in [Starcraft II: A New Challenge for Reinforcement Learning](https://arxiv.org/abs/1708.04782).

## Requirements
* Python 3 (tested with 3.6)
* pysc2 (tested with 2.0.1)
* tensorflow (tested with 1.9.0)
* StarCraft II + Maps

## Usage

### 1. Install Dependencies

To ensure that the version is compatible with the agents in this repository, I recommend using [Pipenv](https://docs.pipenv.org/).

```
$ pip install pipenv

$ git clone https://github.com/rayheberer/SC2Agents.git

$ cd SC2Agents

$ pipenv install
```

### 2. Install StarCraft II

http://us.battle.net/sc2/en/legacy-of-the-void/

PySC2 expects the game to be installed in `~/StarCraftII/`, but this can be overriden by setting the `SC2PATH` environment variable.

### 3. Download Maps

Download the [ladder maps](https://github.com/Blizzard/s2client-proto#downloads)
and the [mini games](https://github.com/deepmind/pysc2/releases/download/v1.0/mini_games.zip)
and extract them to your `StarcraftII/Maps/` directory.

### 4. Train an Agent

```
$ python -m run --map MoveToCheckpoint --agent deepq_agents.DQNMoveOnly
```

This is equivalent to:
```
$ python -m pysc2.bin.agent --map MoveToCheckpoint --agent deepq_agents.DQNMoveOnly
```

However, it is possible to specify agent-specific hyperparameters as flags.

### 5. Evaluate an Agent

```
$ tensorboard --logdir=./tensorboard/deepq
```

```
$ python -m run --map MoveToCheckpoint --agent deepq_agents.DQNMoveOnly --training=False
```

Use the `--save_dir` and `--ckpt_name` flags to specify a TensorFlow checkpoint to read from and write to. By default, an agent will store checkpoints in `./checkpoints/<name-of-agent-class>`.

### 6. Watch a Replay

```
$ python -m pysc2.bin.play --replay <path-to-replay>
```

## Results

Links to pretrained networks and reporting on their results can be found in [results](https://github.com/rayheberer/SC2Agents/tree/master/results).

The following agents are implemented in this repository:

* __DQNMoveOnly__ - a deep q-learner that processes a single screen feature layer through a convolutional neural network and outputs spacial coordinates for the `Move_screen` action.