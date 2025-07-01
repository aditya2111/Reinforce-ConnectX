# 🎯 REINFORCE Agent for ConnectX (Kaggle Environment)

This project implements the **REINFORCE policy gradient algorithm** from scratch using PyTorch to play the **ConnectX** game, hosted on [Kaggle Environments](https://www.kaggle.com/docs/learn/environments/connectx).

The agent learns to play ConnectX via self-play, updates its policy after each episode, and can be evaluated against a random baseline agent.

---

## 🌐 Environment: ConnectX

* **Observation space**: 1D list of length 42 (6x7 board)
* **Action space**: Discrete (column index to drop a piece)
* **Objective**: Get `inarow=4` pieces in a row horizontally, vertically, or diagonally before the opponent.

---

## 🧠 Algorithm: REINFORCE

This project uses the **Monte Carlo REINFORCE** algorithm with:

* Policy network outputting action logits.
* Episode-level reward accumulation.
* Masking of invalid actions (columns already filled).
* Self-play for data collection.
* Reward assigned only at terminal step (win/loss).

---

## 🛠️ Features

Full training pipeline with reward signal
Self-play with alternating players
Board management from 1D format
Invalid action masking
REINFORCE training with policy gradient updates
Win rate evaluation against random agent
Export-ready submission agent for Kaggle

---

## 🧱 Code Structure

### `REINFORCEAgent`

A class encapsulating:

* Policy network (MLP with 2 hidden layers)
* Action selection with masking
* Episode memory (state, action, reward, log\_prob)
* Policy update via policy gradients

### `play_episode(agent, env_config, as_player_1)`

Self-play episode simulation:

* Alternates player turns
* Tracks valid actions
* Assigns reward only at terminal step
* Returns game outcome for learning

### `train_agent(episodes)`

* Trains the agent using self-play
* Updates the policy after each episode
* Tracks rolling win rate

### `create_kaggle_agent(agent)`

* Converts trained PyTorch agent into a Kaggle-compatible inference function.

---

## 🔧 Installation

```bash
pip install 'kaggle-environments>=0.1.6'
pip install torch numpy
```

---

## 🏗️ Training

To train the agent for 10,000 episodes:

```bash
python reinforce_connectx.py
```

* Logs training progress every 1,000 episodes
* Saves model to `reinforce_connectx_model.pth`

---

## 📈 Evaluation

Evaluate the agent’s performance against the built-in random agent:

```bash
# Create evaluation-compatible agent
kaggle_agent = create_kaggle_agent(trained_agent)

# Evaluate
results_p1 = evaluate("connectx", [kaggle_agent, "random"], num_episodes=100)
results_p2 = evaluate("connectx", ["random", kaggle_agent], num_episodes=100)

print(f"Win rate as Player 1: {wins_p1}%")
print(f"Win rate as Player 2: {wins_p2}%")
```

---

## 🧪 Example Output

```
Training REINFORCE agent...
Episode 0, Win Rate (last 100): 0.000
...
Episode 9000, Win Rate (last 100): 0.730
Model saved as 'reinforce_connectx_model.pth'

Testing against random agent...
Win rate as Player 1: 76%
Win rate as Player 2: 69%
Overall win rate: 72.5%
```

---

## 📤 Kaggle Submission

To submit your agent on Kaggle:

1. Save your model: `torch.save(agent.policy_net.state_dict(), 'model.pth')`
2. Export `kaggle_agent()` as a standalone `.py` file.
3. Submit to a [ConnectX competition](https://www.kaggle.com/competitions/connectx) following Kaggle's rules.

---

## 📚 References

* [REINFORCE paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* Sutton & Barto, *Reinforcement Learning: An Introduction*
* [Kaggle ConnectX Environment Docs](https://www.kaggle.com/docs/learn/environments/connectx)

---

## 🧠 Future Improvements

* Add baseline/value function for variance reduction (Actor-Critic)
* Train against stronger opponents
* Curriculum learning: start with simple opponents, then gradually increase difficulty
* Use win probability shaping instead of binary reward

---

## 🤝 Acknowledgments

* [Kaggle Environments](https://github.com/Kaggle/kaggle-environments)
* PyTorch for neural network implementation
