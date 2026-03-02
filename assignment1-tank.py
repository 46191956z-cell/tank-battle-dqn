import sys
sys.path = [p for p in sys.path if "/opt/anaconda3/envs/tank" in p]
print("Current active paths: ", sys.path)

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm

# ===================== Global Configuration =====================
# Game/Training mode switch (disable UI rendering during training)
TRAIN_MODE = False

# Game window and frame rate settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
TRAIN_FPS = 0  # Unlimit frame rate during training to speed up training

# Color configuration (game UI)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)
GREEN = (50, 200, 50)
BLUE = (50, 100, 200)
GRAY = (210, 210, 210)
BG_LIGHT = (245, 247, 250)
PANEL = (225, 230, 235)
SHADOW = (160, 160, 160)
WALL_COLOR = (100, 100, 100)
WALL_BORDER = (70, 70, 70)
TRACK_COLOR = (60, 60, 60)
TURRET_COLOR = (30, 30, 30)

# Game object size and speed parameters
TANK_BODY_SIZE = 42
TANK_TRACK_WIDTH = 8
TURRET_SIZE = 28
BULLET_SIZE = 10
BULLET_SPEED = 10
TANK_SPEED = 5
LIVES = 3  # Initial lives for both tanks

# DQN core hyperparameters (key reinforcement learning settings)
STATE_DIM = 23  # 23-dimensional state space vector
ACTION_DIM = 6  # 6 discrete actions: up/down/left/right/shoot/stop
EPSILON = 1.0   # Initial ε-greedy exploration value
EPSILON_DECAY = 0.995  # ε decay rate (balance exploration/exploitation)
EPSILON_MIN = 0.05     # Minimum ε value (retain small exploration rate)
LEARNING_RATE = 0.0005 # Adam optimizer learning rate
GAMMA = 0.95           # Discount factor (weight future rewards)
BATCH_SIZE = 64        # Experience replay batch size
MEMORY_CAPACITY = 10000# Experience replay buffer capacity
TRAIN_EPISODES = 1000  # Total training episodes
TARGET_UPDATE_FREQ = 30# Target network update frequency (stable training)

# AI difficulty settings (1000 training episodes = maximum difficulty)
BEST_DIFFICULTY = {"name": "Extreme", "ep": 1000, "color": (150, 0, 0)}

# ===================== DQN Network Model =====================
# Use neural network to approximate Q-value function instead of traditional Q-table (adapt to continuous state space)
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Three-layer fully connected network + ReLU activation (introduce non-linearity)
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()          # ReLU activation function
        self.dropout = nn.Dropout(0.1) # Dropout to prevent overfitting

    # Forward propagation: input state → output Q-values for each action
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ===================== Tank Base Class =====================
# General attributes and methods for player/AI tanks
class Tank:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.body_color = color
        self.lives = LIVES
        self.direction = "up"
        self.prev_x = x  # Record previous frame position (for idle penalty)
        self.prev_y = y
        self.no_move_frames = 0  # Idle frame counter

    # Tank movement (with boundary collision detection)
    def move(self, dx, dy):
        self.x = max(TANK_BODY_SIZE//2, min(SCREEN_WIDTH-TANK_BODY_SIZE//2, self.x+dx))
        self.y = max(TANK_BODY_SIZE//2, min(SCREEN_HEIGHT-TANK_BODY_SIZE//2, self.y+dy))

    # Draw tank (body/turret/tracks/lives)
    def draw(self, screen):
        if TRAIN_MODE:
            return
        # Draw tracks
        pygame.draw.rect(screen, TRACK_COLOR, (self.x - TANK_BODY_SIZE//2 - TANK_TRACK_WIDTH, self.y - TANK_BODY_SIZE//2, TANK_TRACK_WIDTH, TANK_BODY_SIZE), border_radius=2)
        pygame.draw.rect(screen, TRACK_COLOR, (self.x + TANK_BODY_SIZE//2, self.y - TANK_BODY_SIZE//2, TANK_TRACK_WIDTH, TANK_BODY_SIZE), border_radius=2)
        # Draw tank body
        pygame.draw.rect(screen, self.body_color, (self.x - TANK_BODY_SIZE//2, self.y - TANK_BODY_SIZE//2, TANK_BODY_SIZE, TANK_BODY_SIZE), border_radius=6)
        pygame.draw.rect(screen, BLACK, (self.x - TANK_BODY_SIZE//2, self.y - TANK_BODY_SIZE//2, TANK_BODY_SIZE, TANK_BODY_SIZE), 2, border_radius=6)
        # Draw turret
        pygame.draw.circle(screen, TURRET_COLOR, (self.x, self.y), TURRET_SIZE//2)
        pygame.draw.circle(screen, BLACK, (self.x, self.y), TURRET_SIZE//2, 2)
        # Draw gun barrel
        gun_length = TANK_BODY_SIZE - 10
        if self.direction == "up":
            pygame.draw.line(screen, BLACK, (self.x, self.y), (self.x, self.y - gun_length), 6)
            pygame.draw.circle(screen, WHITE, (self.x, self.y - gun_length), 3)
        elif self.direction == "down":
            pygame.draw.line(screen, BLACK, (self.x, self.y), (self.x, self.y + gun_length), 6)
            pygame.draw.circle(screen, WHITE, (self.x, self.y + gun_length), 3)
        elif self.direction == "left":
            pygame.draw.line(screen, BLACK, (self.x, self.y), (self.x - gun_length, self.y), 6)
            pygame.draw.circle(screen, WHITE, (self.x - gun_length, self.y), 3)
        elif self.direction == "right":
            pygame.draw.line(screen, BLACK, (self.x, self.y), (self.x + gun_length, self.y), 6)
            pygame.draw.circle(screen, WHITE, (self.x + gun_length, self.y), 3)
        # Draw life count text
        font = pygame.font.SysFont("Arial", 20, bold=True)
        life_text = font.render(str(self.lives), True, WHITE)
        text_rect = life_text.get_rect()
        text_rect.topleft = (self.x + TANK_BODY_SIZE//2 - 20, self.y - TANK_BODY_SIZE//2 - 5)
        pygame.draw.rect(screen, BLACK, (text_rect.x-2, text_rect.y-2, text_rect.width+4, text_rect.height+4))
        screen.blit(life_text, text_rect)

# ===================== Player Tank Class =====================
# Inherit from Tank class, handle keyboard input
class PlayerTank(Tank):
    def __init__(self):
        super().__init__(SCREEN_WIDTH//4, SCREEN_HEIGHT//2, BLUE)

    # Handle WSAD keyboard input
    def handle_input(self, keys):
        dx, dy = 0, 0
        if keys[pygame.K_w]:
            dy = -TANK_SPEED
            self.direction = "up"
        if keys[pygame.K_s]:
            dy = TANK_SPEED
            self.direction = "down"
        if keys[pygame.K_a]:
            dx = -TANK_SPEED
            self.direction = "left"
        if keys[pygame.K_d]:
            dx = TANK_SPEED
            self.direction = "right"
        self.move(dx, dy)

# ===================== AI Tank Class =====================
# Inherit from Tank class, select actions based on DQN
class AITank(Tank):
    def __init__(self):
        super().__init__(3*SCREEN_WIDTH//4, SCREEN_HEIGHT//2, RED)
        self.epsilon = EPSILON  # AI-specific ε value

    # ε-greedy strategy for action selection (balance exploration/exploitation)
    def take_action(self, state, dqn_model, is_training=True):
        # Non-training mode: directly select action with maximum Q-value (exploitation)
        if not is_training:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return torch.argmax(dqn_model(state_tensor)).item()
        # Training mode: random exploration with ε probability, optimal action with 1-ε probability
        if random.random() < self.epsilon:
            action = random.randint(0, ACTION_DIM-1)
        else:
            action = torch.argmax(dqn_model(torch.FloatTensor(state).unsqueeze(0))).item()
        
        # Decay ε value (only in training mode)
        if self.epsilon > EPSILON_MIN and is_training:
            self.epsilon *= EPSILON_DECAY
        
        self.execute_action(action)
        return action

    # Execute selected action (map action index to tank behavior)
    def execute_action(self, action):
        action_map = {
            0: ("up", -TANK_SPEED, 0),
            1: ("down", TANK_SPEED, 0),
            2: ("left", 0, -TANK_SPEED),
            3: ("right", 0, TANK_SPEED),
            4: ("shoot", 0, 0),
            5: ("stop", 0, 0)
        }
        dir_name, dy, dx = action_map[action]
        if dir_name not in ["shoot", "stop"]:
            self.direction = dir_name
            self.move(dx, dy)

# ===================== Bullet Class =====================
class Bullet:
    def __init__(self, x, y, direction, shooter):
        self.x = x
        self.y = y
        self.direction = direction
        self.shooter = shooter  # Mark shooter: player/ai
        self.speed = BULLET_SPEED
        self.active = True
        self.birth_frame = pygame.time.get_ticks() if not TRAIN_MODE else 0

    # Bullet movement
    def move(self):
        if self.direction == "up": self.y -= self.speed
        elif self.direction == "down": self.y += self.speed
        elif self.direction == "left": self.x -= self.speed
        elif self.direction == "right": self.x += self.speed

    # Check if bullet is out of bounds
    def is_out_of_bounds(self):
        return not (0 <= self.x <= SCREEN_WIDTH and 0 <= self.y <= SCREEN_HEIGHT)

    # Draw bullet (with flame effect for first 100ms)
    def draw(self, screen):
        if TRAIN_MODE:
            return
        pygame.draw.circle(screen, (40,40,40), (int(self.x), int(self.y)), BULLET_SIZE//2)
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), BULLET_SIZE//2 - 2)
        if pygame.time.get_ticks() - self.birth_frame < 100:
            pygame.draw.circle(screen, (255,200,0), (int(self.x), int(self.y)), BULLET_SIZE//2 + 3, 2)

# ===================== Game Core Class =====================
# Manage game loop, state, collision, reward, rendering
class Game:
    def __init__(self):
        # Training mode: only initialize pygame, do not create window
        if TRAIN_MODE:
            pygame.init()
            self.screen = None
            self.font = None
            self.small_font = None
        # Play mode: create full game window
        else:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Tank Battle")
            self.font = pygame.font.SysFont("Arial", 28, bold=True)
            self.small_font = pygame.font.SysFont("Arial", 20)
        
        self.clock = pygame.time.Clock()
        # Game wall obstacles (thick walls: 3 HP, thin walls: 2 HP)
        self.walls = [
            {"rect": pygame.Rect(200, 150, 60, 300), "hp": 3},
            {"rect": pygame.Rect(540, 150, 60, 300), "hp": 3},
            {"rect": pygame.Rect(320, 450, 160, 60), "hp": 2},
            {"rect": pygame.Rect(320, 90, 160, 60), "hp": 2},
        ]
        self.reset()

    # Reset game state (new episode/training)
    def reset(self):
        self.player = PlayerTank()
        self.ai_tank = AITank()
        self.bullets = []
        self.score = 0
        self.game_over = False
        self.frame_count = 0
        self.current_reward = 0
        self.current_action = "None"
        # Reset wall HP
        for wall in self.walls:
            wall["hp"] = 3 if wall["rect"].width == 60 else 2

    # Get game state (encoded as 23-dimensional vector for DQN input)
    def get_state(self):
        # Direction encoding: up=0/down=1/left=2/right=3
        player_dir = {"up":0,"down":1,"left":2,"right":3}[self.player.direction]
        ai_dir = {"up":0,"down":1,"left":2,"right":3}[self.ai_tank.direction]
        
        # Player state (normalized position, numerically encoded direction, normalized lives)
        p_state = [
            self.player.x/SCREEN_WIDTH,
            self.player.y/SCREEN_HEIGHT,
            player_dir/3,
            self.player.lives/LIVES
        ]
        # AI state (normalized position, numerically encoded direction, normalized lives)
        ai_state = [
            self.ai_tank.x/SCREEN_WIDTH,
            self.ai_tank.y/SCREEN_HEIGHT,
            ai_dir/3,
            self.ai_tank.lives/LIVES
        ]
        # Bullet state (up to 5 bullets, fill unused dimensions with 0)
        bullet_state = []
        sorted_bullets = sorted(self.bullets, key=lambda b: np.hypot(b.x-self.ai_tank.x, b.y-self.ai_tank.y))[:5]
        for b in sorted_bullets:
            b_dir = {"up":0,"down":1,"left":2,"right":3}[b.direction]
            bullet_state.extend([b.x/SCREEN_WIDTH, b.y/SCREEN_HEIGHT, b_dir/3])
        bullet_state += [0.0] * (15 - len(bullet_state))
        
        # Return 23-dimensional normalized state vector
        return np.array(p_state + ai_state + bullet_state, dtype=np.float32)

    # Reward function (core RL design: guide AI to learn optimal strategies)
    def calculate_reward(self, ai_action):
        reward = 0.05  # Basic survival reward per frame
        
        # Penalty: AI tank near map edge (avoid passive corner-hiding behavior)
        if self.ai_tank.x < 80 or self.ai_tank.x > SCREEN_WIDTH-80 or self.ai_tank.y < 80 or self.ai_tank.y > SCREEN_HEIGHT-80:
            reward -= 0.8
        
        # Reward: AI hits player tank (core victory goal)
        hit_player = any(b.shooter=="ai" and self.check_collision(b, self.player) for b in self.bullets)
        if hit_player:
            reward += 50.0
        
        # Penalty: AI hit by player tank (encourage evasion of enemy bullets)
        hit_ai = any(b.shooter=="player" and self.check_collision(b, self.ai_tank) for b in self.bullets)
        if hit_ai:
            reward -= 40.0
        
        # Reward/Penalty: Shooting control (avoid meaningless random shooting)
        ai_bullets = len([b for b in self.bullets if b.shooter=="ai"])
        if ai_action == 4:
            if ai_bullets <= 1:  # Reward for reasonable shooting (≤1 bullet on field)
                reward += 0.5
            else:  # Penalty for random shooting (>1 bullet on field)
                reward -= 1.0
        
        # Penalty: Player bullets close to AI (threat avoidance)
        for b in self.bullets:
            if b.shooter == "player":
                distance = np.hypot(b.x-self.ai_tank.x, b.y-self.ai_tank.y)
                if distance < TANK_BODY_SIZE * 2:
                    reward -= 0.5
        
        # Reward: AI close to player (encourage pursuit)
        dist_to_player = np.hypot(self.ai_tank.x-self.player.x, self.ai_tank.y-self.player.y)
        if dist_to_player < 300:
            reward += (300 - dist_to_player) / 100
        
        # Penalty: AI stays stationary for 15 consecutive frames (encourage active exploration)
        if abs(self.ai_tank.x - self.ai_tank.prev_x) < 1 and abs(self.ai_tank.y - self.ai_tank.prev_y) < 1:
            self.ai_tank.no_move_frames += 1
            if self.ai_tank.no_move_frames > 15:
                reward -= 1.0
        else:
            self.ai_tank.no_move_frames = 0
        
        # Update AI position and current reward
        self.ai_tank.prev_x, self.ai_tank.prev_y = self.ai_tank.x, self.ai_tank.y
        self.current_reward = reward
        return reward

    # Collision detection (bullet vs tank)
    def check_collision(self, bullet, tank):
        bullet_rect = pygame.Rect(bullet.x-BULLET_SIZE//2, bullet.y-BULLET_SIZE//2, BULLET_SIZE, BULLET_SIZE)
        tank_rect = pygame.Rect(tank.x-TANK_BODY_SIZE//2, tank.y-TANK_BODY_SIZE//2, TANK_BODY_SIZE, TANK_BODY_SIZE)
        return bullet_rect.colliderect(tank_rect)

    # Fire bullet (max 2 bullets per shooter to avoid spamming)
    def fire_bullet(self, shooter):
        tank = self.player if shooter == "player" else self.ai_tank
        if len([b for b in self.bullets if b.shooter == shooter]) < 2:
            self.bullets.append(Bullet(tank.x, tank.y, tank.direction, shooter))

    # Update game state (shooting/bullet movement/collision/game over)
    def update(self, ai_action=None, event_list=None):
        if self.game_over: return
        event_list = event_list or []
        
        # Player shooting (space bar)
        for e in event_list:
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                self.fire_bullet("player")
        
        # AI shooting (action index 4)
        if ai_action == 4:
            self.fire_bullet("ai")
            self.current_action = "Shoot"
        else:
            self.current_action = ["Up","Down","Left","Right","Shoot","Stop"][ai_action] if ai_action is not None else "None"
        
        # Bullet movement and collision detection
        for b in self.bullets[:]:
            b.move()
            wall_hit = False
            # Bullet hits wall (reduce wall HP)
            for wall in self.walls:
                if wall["hp"] > 0 and b.active and wall["rect"].collidepoint(b.x, b.y):
                    wall["hp"] -= 1
                    self.bullets.remove(b)
                    b.active = False
                    wall_hit = True
                    break
            if wall_hit: continue
            
            # Bullet hits player tank
            if b.shooter == "ai" and self.check_collision(b, self.player):
                self.player.lives -= 1
                self.bullets.remove(b)
                if self.player.lives <= 0:
                    self.game_over = True
            # Bullet hits AI tank
            elif b.shooter == "player" and self.check_collision(b, self.ai_tank):
                self.ai_tank.lives -= 1
                self.bullets.remove(b)
                if self.ai_tank.lives <= 0:
                    self.game_over = True
            # Bullet out of bounds
            elif b.is_out_of_bounds():
                self.bullets.remove(b)
        
        self.frame_count += 1

    # Draw game UI (fixed: all screen → self.screen)
    def draw_ui(self):
        if TRAIN_MODE or not self.screen:
            return
        # Draw background and grid
        self.screen.fill(BG_LIGHT)
        for y in range(0, SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, (235,235,235), (0, y), (SCREEN_WIDTH, y), 1)
        for x in range(0, SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, (235,235,235), (x, 0), (x, SCREEN_HEIGHT), 1)
        
        # Draw walls (with HP bar)
        for wall in self.walls:
            if wall["hp"] > 0:
                pygame.draw.rect(self.screen, WALL_COLOR, wall["rect"])
                pygame.draw.rect(self.screen, WALL_BORDER, wall["rect"], 3)
                pygame.draw.rect(self.screen, (220,220,220), (wall["rect"].x, wall["rect"].y-10, wall["rect"].width, 6))
                pygame.draw.rect(self.screen, (255,0,0), (wall["rect"].x, wall["rect"].y-10, wall["rect"].width*(wall["hp"]/3), 6))
        
        # Draw UI panel
        pygame.draw.rect(self.screen, PANEL, (10,10, SCREEN_WIDTH-20, 80), border_radius=10)
        pygame.draw.rect(self.screen, SHADOW, (10,10, SCREEN_WIDTH-20, 80), 2, border_radius=10)
        
        # Draw tanks and bullets
        self.player.draw(self.screen)
        self.ai_tank.draw(self.screen)
        for b in self.bullets:
            b.draw(self.screen)
        
        # Draw life bars
        pygame.draw.rect(self.screen, (220,220,220), (30,25,150,25), border_radius=6)
        pygame.draw.rect(self.screen, BLUE, (30,25,self.player.lives*50,25), border_radius=6)
        pygame.draw.rect(self.screen, (220,220,220), (SCREEN_WIDTH-180,25,150,25), border_radius=6)
        pygame.draw.rect(self.screen, RED, (SCREEN_WIDTH-180,25,self.ai_tank.lives*50,25), border_radius=6)
        
        # Draw UI text
        self.screen.blit(self.small_font.render("PLAYER", True, BLACK), (30,55))
        self.screen.blit(self.small_font.render("AI TANK", True, BLACK), (SCREEN_WIDTH-180,55))
        self.screen.blit(self.font.render(f"Score: {self.score}", True, (60,60,60)), (200,25))
        self.screen.blit(self.font.render(f"Frames: {self.frame_count}", True, (60,60,60)), (400,25))
        self.screen.blit(self.small_font.render(f"Reward: {self.current_reward:.1f}", True, GREEN), (200,55))
        self.screen.blit(self.small_font.render(f"Action: {self.current_action}", True, (30,30,30)), (350,55))
        
        # Draw game over screen
        if self.game_over:
            mask = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            mask.set_alpha(128)
            mask.fill(BLACK)
            self.screen.blit(mask, (0,0))
            self.screen.blit(self.font.render("GAME OVER", True, (220,0,0)), (SCREEN_WIDTH//2-80, 260))
            self.screen.blit(self.small_font.render("R=Restart | Q=Quit", True, WHITE), (SCREEN_WIDTH//2-100, 300))
        
        pygame.display.flip()

# ===================== Play Mode =====================
def play_game():
    global TRAIN_MODE
    TRAIN_MODE = False
    
    # Reset SDL environment (enable UI)
    os.environ.pop('SDL_VIDEODRIVER', None)
    os.environ.pop('SDL_AUDIODRIVER', None)
    pygame.quit()
    pygame.init()
    
    # Initialize game and DQN model
    game = Game()
    dqn_model = DQN(STATE_DIM, ACTION_DIM)
    model_path = f"models/dqn_tank_ep{BEST_DIFFICULTY['ep']}.pth"
    
    # Load trained model (handle model not found)
    try:
        dqn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        dqn_model.eval()  # Inference mode (disable dropout)
        print(f"✅ Successfully loaded {BEST_DIFFICULTY['name']} difficulty model: {model_path}")
    except FileNotFoundError:
        print(f"❌ {BEST_DIFFICULTY['name']} difficulty model not found! Please train the model first.")
        # Load latest model (if available)
        model_files = [f for f in os.listdir("models") if f.startswith("dqn_tank_ep")] if os.path.exists("models") else []
        if model_files:
            latest_model = max(model_files, key=lambda x: int(x.split("ep")[1].split(".pth")[0]))
            dqn_model.load_state_dict(torch.load(f"models/{latest_model}", map_location=torch.device('cpu')))
            dqn_model.eval()
            print(f"💡 Loaded latest model: {latest_model}")
        else:
            print("❌ No trained models found!")
            return

    # Inference mode ε value (small random exploration)
    game.ai_tank.epsilon = 0.3
    print(f"🎮 {BEST_DIFFICULTY['name']} difficulty game started! WSAD to move, SPACE to shoot, R to restart, Q to quit")

    # Game main loop
    while True:
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    game.reset()
                if e.key == pygame.K_q:
                    pygame.quit()
                    return
        
        # AI select action (inference mode)
        state = game.get_state()
        action = game.ai_tank.take_action(state, dqn_model, is_training=False)
        game.ai_tank.execute_action(action)
        
        # Player input handling
        game.player.handle_input(pygame.key.get_pressed())
        
        # Update game state and render UI
        game.update(action, events)
        game.calculate_reward(action)
        game.draw_ui()
        game.clock.tick(FPS)

# ===================== Training Mode =====================
def train_dqn():
    global TRAIN_MODE
    TRAIN_MODE = True  # Enable training mode (disable UI)
    
    # Force SDL virtual driver (UI-less training)
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    # Create reward log file
    reward_log = open("reward_log.csv", "w")
    reward_log.write("Episode,Total Reward,Epsilon,Avg Loss\n")
    
    # Create model save directory
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Initialize game, DQN model, target network
    game = Game()
    dqn_model = DQN(STATE_DIM, ACTION_DIM)       # Policy network (frequent updates)
    target_model = DQN(STATE_DIM, ACTION_DIM)    # Target network (slow updates)
    target_model.load_state_dict(dqn_model.state_dict())  # Synchronize initial weights
    
    # Optimizer and loss function
    optimizer = optim.Adam(dqn_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Experience replay buffer
    memory = deque(maxlen=MEMORY_CAPACITY)
    
    # Training progress bar
    pbar = tqdm(range(TRAIN_EPISODES), desc=f"Training {BEST_DIFFICULTY['name']} difficulty AI (1000 episodes)")

    # Training main loop
    for episode in pbar:
        game.reset()  # Reset game state
        total_reward = 0  # Total reward for current episode
        total_loss = 0
        loss_count = 0
        state = game.get_state()  # Initial state

        # Single episode training loop (until game over)
        while not game.game_over:
            events = []
            # AI select action
            action = game.ai_tank.take_action(state, dqn_model)
            
            # Training mode: simulate random player movement (help AI learn)
            if TRAIN_MODE:
                dx, dy = 0, 0
                direction = random.choice(["up", "down", "left", "right", "stop", "stop"])
                if direction == "up":
                    dy = -TANK_SPEED * 0.5
                    game.player.direction = "up"
                elif direction == "down":
                    dy = TANK_SPEED * 0.5
                    game.player.direction = "down"
                elif direction == "left":
                    dx = -TANK_SPEED * 0.5
                    game.player.direction = "left"
                elif direction == "right":
                    dx = TANK_SPEED * 0.5
                    game.player.direction = "right"
                game.player.move(dx, dy)
            
            # Update game state, get reward and next state
            game.update(action, events)
            next_state = game.get_state()
            reward = game.calculate_reward(action)
            total_reward += reward
            
            # Store experience to buffer: (state, action, reward, next state, game over flag)
            memory.append((state, action, reward, next_state, game.game_over))
            state = next_state

            # Experience replay (train when buffer has enough samples)
            if len(memory) >= BATCH_SIZE:
                # Random sample batch (break temporal correlation)
                batch = random.sample(memory, BATCH_SIZE)
                states = torch.FloatTensor(np.array([x[0] for x in batch]))
                actions = torch.LongTensor(np.array([x[1] for x in batch])).unsqueeze(1)
                rewards = torch.FloatTensor(np.array([x[2] for x in batch]))
                next_states = torch.FloatTensor(np.array([x[3] for x in batch]))
                dones = torch.FloatTensor(np.array([x[4] for x in batch]))

                # Calculate current Q-value: Q(s,a)
                current_q = dqn_model(states).gather(1, actions).squeeze(1)
                # Calculate target Q-value: r + γ * max(Q'(s',a')) (only r if game over)
                next_q = target_model(next_states).max(1)[0]
                target_q = rewards + GAMMA * next_q * (1 - dones)

                # Model training
                loss = criterion(current_q, target_q)
                optimizer.zero_grad()  # Clear gradients
                loss.backward()        # Backward propagation
                optimizer.step()       # Update weights
                total_loss += loss.item()
                loss_count += 1

            game.clock.tick(TRAIN_FPS)

        # Update progress bar info
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        pbar.set_postfix({
            "Total Reward": f"{total_reward:.2f}",
            "Epsilon": f"{game.ai_tank.epsilon:.3f}",
            "Avg Loss": f"{avg_loss:.4f}"
        })
        
        # Write to reward log
        reward_log.write(f"{episode+1},{total_reward:.2f},{game.ai_tank.epsilon:.3f},{avg_loss:.4f}\n")
        
        # Update target network
        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            target_model.load_state_dict(dqn_model.state_dict())
        
        # Save final model
        model_path = f"models/dqn_tank_ep{BEST_DIFFICULTY['ep']}.pth"
        if (episode + 1) == TRAIN_EPISODES:
            torch.save(dqn_model.state_dict(), model_path)

    # Close log file
    reward_log.close()
    
    # Save best model
    torch.save(dqn_model.state_dict(), "models/dqn_tank_best.pth")
    print(f"\n✅ {BEST_DIFFICULTY['name']} difficulty AI training completed! Model saved to {model_path}")
    print(f"✅ Reward log saved to reward_log.csv (open with Excel)")
    pygame.quit()

# ===================== Main Function =====================
if __name__ == "__main__":
    print("===== Tank Battle Game ======")
    print("Please select an operation:")
    print("1) Train DQN model (1000 episodes, ~10 minutes)")
    print("2) Play against trained AI")
    choice = input("Enter selection (1/2): ").strip()
    if choice == "1":
        train_dqn()
    elif choice == "2":
        play_game()
    else:
        print("Invalid input, defaulting to training AI...")
        train_dqn()
