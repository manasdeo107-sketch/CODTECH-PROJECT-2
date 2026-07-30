# Project Report — Reinforcement Learning for Games

**Intern ID:** CITS7621
**Intern Name:** Manas Deo
**Project Name:** Reinforcement Learning for Games
**Program Duration:** 4-Week Internship Program
**Domain:** Machine Learning (Reinforcement Learning)

---

## Project Scope

The goal of this project is to design and implement a Reinforcement
Learning (RL) agent that learns to play the game of **Tic-Tac-Toe**
through trial and error, without being given any labeled dataset.
Instead of learning from static examples (as in supervised learning),
the agent learns entirely from its own experience — playing thousands
of games, receiving rewards for wins/draws/losses, and gradually
improving its strategy using **Q-Learning**.

The system covers the complete RL pipeline: environment design →
agent design → training via self-play → evaluation → interactive
human-vs-agent play, and is packaged as a clean, reusable, GitHub-ready
project.

---

## Week-wise Breakdown

### Week 1 — Problem Understanding & Environment Design
- Studied core Reinforcement Learning concepts: agents, environments,
  states, actions, rewards, policies, and the exploration/exploitation
  trade-off.
- Selected Tic-Tac-Toe as the game environment — simple enough to
  train quickly with tabular methods, but rich enough to demonstrate
  real strategic learning.
- Built `src/environment.py` from scratch: board representation,
  valid-move detection, win/draw checking, and a step function
  returning `(next_state, reward, done, info)` following the standard
  RL interface.

### Week 2 — Agent Design (Q-Learning)
- Built `src/q_learning_agent.py` implementing tabular Q-Learning:
  - A Q-table mapping (state, action) pairs to expected future reward.
  - An epsilon-greedy action selection policy for balancing
    exploration vs exploitation.
  - The Bellman equation update rule for learning from experience.
  - Epsilon decay so the agent explores heavily early on and exploits
    its learned knowledge later in training.

### Week 3 — Training & Evaluation
- Built `src/train_agent.py` to train the agent over 50,000 episodes
  against a random-move opponent.
- Logged every episode's reward, result, and epsilon value to
  `outputs/reports/training_log.csv` (this project's "dataset" is
  generated dynamically through gameplay, unlike a static CSV).
- Plotted the moving-average reward curve and the win/draw/loss rate
  over training blocks to visualize learning progress.
- Built `src/evaluate_agent.py` to test the fully-trained agent
  (greedy, no exploration) over 2,000 evaluation games.

### Week 4 — Interactive Play & Documentation
- Built `src/play_game.py`, an interactive CLI game where a human can
  play Tic-Tac-Toe against the trained agent in real time.
- Organized the entire project into a clean, GitHub-ready folder
  structure with a full README, requirements file, and this report.

---

## Results Summary

After training for 50,000 episodes against a random opponent, and
evaluating over 2,000 fully-greedy games:

| Metric | Result |
|---|---|
| Win Rate | ~98.8% |
| Draw Rate | ~1.2% |
| Loss Rate | ~0.0% |
| Learned Q-table size | ~7,500 state-action pairs |

The agent learned to reliably win or draw against a random opponent,
including correctly blocking opponent threats and taking winning
moves when available — all without ever being explicitly told the
rules of strategy, only the rules of the game itself (via rewards).

---

## Key Learnings

- Reinforcement Learning does not need a labeled dataset — the agent
  generates its own training signal (reward) purely through
  interaction with the environment.
- The epsilon-greedy strategy is essential: too little exploration
  and the agent gets stuck with a suboptimal policy; too much and it
  never converges to a strong strategy in a reasonable time.
- Because Tic-Tac-Toe has a small, finite state space, tabular
  Q-Learning (a simple lookup table) is sufficient — larger games
  would need function approximation (e.g. Deep Q-Networks).

## Future Improvements

- Train the agent via **self-play** (playing against a copy of
  itself) instead of a random opponent, which typically produces a
  stronger, more general strategy.
- Extend to more complex games (e.g. Connect Four) where a Deep
  Q-Network (DQN) would be needed instead of a plain Q-table.
- Add a simple web or GUI interface (instead of CLI) for playing
  against the trained agent.
