# StarCraft II Agents

These agents interface directly with Deepmind's [PySC2](https://github.com/deepmind/pysc2) API.

## Usage

### 1. Get PySC2 and other Dependencies

To ensure that the version is compatible with the agents in this repository, I recommend using [Pipenv](https://docs.pipenv.org/).

```
$ pip install pipenv

$ git clone https://github.com/rayheberer/SC2Agents.git

$ cd SC2Agents

$ pipenv install
```

### 2. Install StarCraft II

http://us.battle.net/sc2/en/legacy-of-the-void/

### 3. Download Maps

Download the [ladder maps](https://github.com/Blizzard/s2client-proto#downloads)
and the [mini games](https://github.com/deepmind/pysc2/releases/download/v1.0/mini_games.zip)
and extract them to your `StarcraftII/Maps/` directory.

### 4. Train an Agent

```
$ python -m pysc2.bin.agent --map CollectMineralShards --agent deepq_agents.DQNMoveOnly
```

### 5. View Metrics on Tensorboard

```
$ tensorboard --logdir=./tensorboard/deepq
```

### 6. Watch a Replay

`$ python -m pysc2.bin.play --replay <path-to-replay>`