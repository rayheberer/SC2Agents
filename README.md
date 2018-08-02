# StarCraft II Agents

These agents interface directly with Deepmind's [PySC2](https://github.com/deepmind/pysc2) API

## Usage

### 1. Get PySC2

`$ pip install pysc2`

### 2. Install StarCraft II

http://us.battle.net/sc2/en/legacy-of-the-void/

### 3. Download Maps

Download the [ladder maps](https://github.com/Blizzard/s2client-proto#downloads)
and the [mini games](https://github.com/deepmind/pysc2/releases/download/v1.0/mini_games.zip)
and extract them to your `StarcraftII/Maps/` directory.

### 4. Run an Agent

`$ git clone https://github.com/rayheberer/SC2Agents`

`$ python -m pysc2.bin.agent --map CollectMineralShards --agent SC2Agents.random.QueueRandomMovements`

### 5. Watch a Replay

`$ python -m pysc2.bin.play --replay <path-to-replay>`