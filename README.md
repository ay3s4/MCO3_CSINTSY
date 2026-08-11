# CatBot

Reinforcement learning bot that catches cats on an 8x8 grid, for CSINTSY MCO3.
The learning algorithm used is tabular Q-learning, implemented in `training.py`.

## Setup

Python 3.12 is required for the project.

Install the required packages:

```bash
py -3.12 -m pip install -r requirements.txt
```

Make sure you are inside the `catbot/` directory before running the commands below:

```bash
cd catbot
```

## Launch

| Command                          | What it does                    |
| -------------------------------- | ------------------------------- |
| `py -3.12 play.py --cat mittens` | Play the game manually          |
| `py -3.12 bot.py --cat paotsin`  | Train the bot and watch it play |

You can replace `--cat` with:

```text
batmeow
mittens
paotsin
peekaboo
squiddyboi
```

Press **Q** to quit the game.

## Evaluation

Run:

```bash
py -3.12 evaluation.py
```

The evaluation tests the trained bot against different cat behaviors and provides results such as:

* Number of successful catches
* Success rate
* Average number of moves
* Worst number of moves

These results can be used for the performance evaluation in the report.

## Project Files

* `bot.py` – Trains the bot and runs the trained policy.
* `training.py` – Contains the Q-learning implementation.
* `evaluation.py` – Evaluates the bot's performance.
* `utility.py` – Contains supporting functions.
* `cat_env.py` – Provides the game environment.
* `play.py` – Runs the game.
* `images/` – Contains the game sprites.
* `requirements.txt` – Lists the required Python packages.
