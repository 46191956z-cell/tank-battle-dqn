# DQN-Based Tank Battle Game
A 2D tank battle game implemented with Deep Q-Network (DQN) reinforcement learning, designed for the CDS524 Reinforcement Learning course assignment. The AI agent autonomously learns combat strategies (movement, shooting, evasion) to confront human players, demonstrating the application of DQN in sequential decision-making tasks.

## 📋 Project Overview
This project builds a fully functional 2D tank battle game using Python, with:
- **Pygame**: Constructs the game interface, physical environment, and interactive UI.
- **PyTorch**: Implements the DQN neural network for AI training (experience replay, ε-greedy exploration, target network).
- **Google Colab**: Enables cloud-based code debugging and model training (CPU-only, ~10 minutes for 1000 episodes).
- **GitHub**: Hosts all source code, training logs, and documentation for easy access and reproducibility.

## 🔧 Environment Setup
### Prerequisites
- Python 3.8+ (compatible with Colab/本地环境)
- Required dependencies (install via pip):
```bash
pip install pygame torch numpy tqdm
Colab Special Setup
Add the following code at the top of the Colab notebook to install dependencies and configure the SDL virtual driver (for headless training):
python
运行
!pip install -q pygame torch numpy tqdm
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
🎮 How to Run
1. Train the DQN Model
Local Run: Execute assignment1-tank.py and select option 1 (train AI for 1000 episodes).
Colab Run: Open the .ipynb notebook and run all cells (auto-enables training mode).
Outputs:
Trained model weights saved to models/dqn_tank_ep1000.pth
Training log (reward/epsilon/loss) saved to reward_log.csv
2. Play Against Trained AI
Run assignment1-tank.py and select option 2 (play mode).
Controls:
WSAD: Move the blue player tank (left side)
SPACE: Shoot bullets
R: Restart the game
Q: Quit the game
📁 File Structure
plaintext
tank-battle-dqn/
├── Colab_Assignment1_Yang_Peilun.ipynb  # Colab-adapted notebook (cloud training)
├── assignment1-tank.py                   # Local runnable full code (play/train mode)
├── reward_log.csv                        # Training log (episode/reward/epsilon/loss)
├── models/                               # Trained DQN model weights (auto-created)
│   ├── dqn_tank_ep1000.pth              # Final model (1000 episodes)
│   └── dqn_tank_best.pth                 # Backup of the best model
└── README.md                             # Project documentation (this file)
📊 Experimental Results
Training Performance
Exploration Stage (Episodes 1-200): High ε (1.0 → 0.368) leads to random AI behavior, reward fluctuates drastically.
Convergence Stage (Episodes 200-1000): ε decays to 0.05, reward stabilizes at 0-2000 (AI masters survival/attack/evasion).
Key Metrics:
Average loss decreases from ~0.8 to ~0.1 (Q-value prediction accuracy improves).
AI win rate reaches ~70% against human players after full training.
Visualization
Game Interface: The UI displays real-time tank health, reward values, AI actions, and frame count (see demo video).
Reward Curve: reward_log.csv can be visualized with Excel/Matplotlib to show training convergence (included in the report).
🎯 Core DQN Implementation
Network Architecture
Input Layer: 23-dimensional state vector (tank position/direction/health + bullet states)
Hidden Layers: 128 → 64 neurons (ReLU activation + Dropout 0.1 to prevent overfitting)
Output Layer: 6-dimensional Q-values (up/down/left/right/shoot/stop)
Key RL Techniques
Technique	Purpose
ε-greedy Exploration	Balance exploration (random actions) and exploitation (optimal actions)
Experience Replay	Break temporal correlation in training data (buffer size: 10,000)
Target Network	Stabilize training (update every 30 episodes)
Dual-Incentive Reward	Guide AI to prioritize survival, attack, and avoid passive behavior
📈 Project Deliverables
Colab Code: https://colab.research.google.com/drive/1PyoarTR7klvltoDCW9x0MFUzgrs-shpK?usp=drive_link
GitHub Repository: https://github.com/46191956z-cell/tank-battle-dqn.git
Assignment Report: https://github.com/46191956z-cell/tank-battle-dqn.git
🚫 Challenges & Solutions
Challenge	Solution
Imbalanced Reward Function	Adjust reward weights (reduce attack reward, increase evasion penalty)
Training Oscillations	Introduce target network to calculate stable Q-values
UI Slowdown in Training	Add TRAIN_MODE to disable Pygame rendering (SDL virtual driver)
🙏 Acknowledgements
Course materials for reinforcement learning fundamentals
Pygame official documentation for game development
PyTorch tutorials for DQN implementation
📝 License
This project is for educational purposes only (course assignment) and is not intended for commercial use.
