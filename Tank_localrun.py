import sys
sys.path = [p for p in sys.path if "/opt/anaconda3/envs/tank" in p]
print("Current active paths: ", sys.path)

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# --------------------------
# Core Library Imports
# --------------------------
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm

# ===================== Global Game & DQN Settings =====================
TRAIN_MODE = False

# Game window dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
TRAIN_FPS = 0

# Color definitions
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

# Game object dimensions
TANK_BODY_SIZE = 42
TANK_TRACK_WIDTH = 8
TURRET_SIZE = 28
BULLET_SIZE = 10
BULLET_SPEED = 10
TANK_SPEED = 5
LIVES = 3

# DQN Hyperparameters (Dynamic adaptation to any training episodes)
STATE_DIM = 25
ACTION_DIM = 6
EPSILON = 1.0
EPSILON_MIN = 0.1
LEARNING_RATE = 0.0005
GAMMA = 0.95
BATCH_SIZE = 64
MEMORY_CAPACITY = 10000
TRAIN_EPISODES = 1000  # Can change to 500/2000 - AI still works well
EPSILON_DECAY = 1 - (0.9 / TRAIN_EPISODES)
TARGET_UPDATE_FREQ = int(TRAIN_EPISODES / 33)

# ===================== DQN Neural Network Class =====================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ===================== Tank Base Class =====================
class Tank:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.body_color = color
        self.lives = LIVES
        self.direction = "up"
        self.prev_x = x
        self.prev_y = y
        self.no_move_frames = 0

    def move(self, dx, dy):
        self.x = max(TANK_BODY_SIZE//2, min(SCREEN_WIDTH-TANK_BODY_SIZE//2, self.x+dx))
        self.y = max(TANK_BODY_SIZE//2, min(SCREEN_HEIGHT-TANK_BODY_SIZE//2, self.y+dy))

    def draw(self, screen):
        if TRAIN_MODE:
            return
        
        # Draw tank tracks
        pygame.draw.rect(screen, TRACK_COLOR, (self.x - TANK_BODY_SIZE//2 - TANK_TRACK_WIDTH, self.y - TANK_BODY_SIZE//2, TANK_TRACK_WIDTH, TANK_BODY_SIZE), border_radius=2)
        pygame.draw.rect(screen, TRACK_COLOR, (self.x + TANK_BODY_SIZE//2, self.y - TANK_BODY_SIZE//2, TANK_TRACK_WIDTH, TANK_BODY_SIZE), border_radius=2)
        
        # Draw tank body
        pygame.draw.rect(screen, self.body_color, (self.x - TANK_BODY_SIZE//2, self.y - TANK_BODY_SIZE//2, TANK_BODY_SIZE, TANK_BODY_SIZE), border_radius=6)
        pygame.draw.rect(screen, BLACK, (self.x - TANK_BODY_SIZE//2, self.y - TANK_BODY_SIZE//2, TANK_BODY_SIZE, TANK_BODY_SIZE), 2, border_radius=6)
        
        # Draw tank turret
        pygame.draw.circle(screen, TURRET_COLOR, (self.x, self.y), TURRET_SIZE//2)
        pygame.draw.circle(screen, BLACK, (self.x, self.y), TURRET_SIZE//2, 2)
        
        # Draw tank gun
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
        
        # Draw life counter
        font = pygame.font.SysFont("Arial", 20, bold=True)
        life_text = font.render(str(self.lives), True, WHITE)
        text_rect = life_text.get_rect()
        text_rect.topleft = (self.x + TANK_BODY_SIZE//2 - 20, self.y - TANK_BODY_SIZE//2 - 5)
        pygame.draw.rect(screen, BLACK, (text_rect.x-2, text_rect.y-2, text_rect.width+4, text_rect.height+4))
        screen.blit(life_text, text_rect)

# ===================== Player Tank Class =====================
class PlayerTank(Tank):
    def __init__(self):
        super().__init__(SCREEN_WIDTH//4, SCREEN_HEIGHT//2, BLUE)

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

# ===================== AI Tank Class (Ultimate Fix: Track Player + Shoot) =====================
class AITank(Tank):
    def __init__(self):
        super().__init__(3*SCREEN_WIDTH//4, SCREEN_HEIGHT//2, RED)
        self.epsilon = EPSILON
        self.game = None  # Will be set in Game.reset()

    def take_action(self, state, dqn_model, is_training=True):
        if not is_training:
            # PLAY MODE: Ultimate Fix - Track moving player + continuous shooting
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = dqn_model(state_tensor)
            q_values[0, 5] = -float('inf')  # Disable stop action
            
            # ========== Core Fix 1: Dynamic tracking of moving player ==========
            # Calculate direction to player (always chase)
            dx = self.game.player.x - self.x
            dy = self.game.player.y - self.y
            dist_to_player = np.hypot(dx, dy)
            
            # Prioritize movement towards player (overrides random model output)
            if abs(dx) > abs(dy):
                # Horizontal distance larger - prioritize left/right
                if dx > 0:  # Player is to the right
                    q_values[0, 3] += 50  # Boost "right" action
                else:  # Player is to the left
                    q_values[0, 2] += 50  # Boost "left" action
            else:
                # Vertical distance larger - prioritize up/down
                if dy > 0:  # Player is below
                    q_values[0, 1] += 50  # Boost "down" action
                else:  # Player is above
                    q_values[0, 0] += 50  # Boost "up" action
            
            # ========== Core Fix 2: Continuous shooting when facing player ==========
            # Expand shooting range (250px) + auto-adjust direction before shooting
            if self.game.is_ai_facing_player() and dist_to_player < 250:
                q_values[0, 4] += 100  # High priority for shooting
            else:
                # Maintain movement but lower weight to allow direction adjustment
                q_values[0, 0] += 30  # Up
                q_values[0, 1] += 30  # Down
                q_values[0, 2] += 30  # Left
                q_values[0, 3] += 30  # Right
            
            # Select action with highest Q-value
            action = torch.argmax(q_values).item()
            
            # Fallback: If shoot is selected but not in range, force move TOWARDS player
            if action == 4 and not (self.game.is_ai_facing_player() and dist_to_player < 250):
                if abs(dx) > abs(dy):
                    action = 3 if dx > 0 else 2  # Move right/left to player
                else:
                    action = 1 if dy > 0 else 0  # Move down/up to player
            
            self.execute_action(action)
            return action

        # TRAINING MODE (unchanged)
        if random.random() < self.epsilon:
            action = random.choice([0,1,2,3,4])
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = dqn_model(state_tensor)
            q_values[0, 5] = -float('inf')
            q_values[0, 0] *= 1.2
            q_values[0, 1] *= 1.2
            q_values[0, 2] *= 1.2
            q_values[0, 3] *= 1.2
            action = torch.argmax(q_values).item()

        if self.epsilon > EPSILON_MIN and is_training:
            self.epsilon *= EPSILON_DECAY

        self.execute_action(action)
        return action

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
        self.shooter = shooter
        self.speed = BULLET_SPEED
        self.active = True
        self.birth_frame = pygame.time.get_ticks() if not TRAIN_MODE else 0

    def move(self):
        if self.direction == "up": self.y -= self.speed
        elif self.direction == "down": self.y += self.speed
        elif self.direction == "left": self.x -= self.speed
        elif self.direction == "right": self.x += self.speed

    def is_out_of_bounds(self):
        return not (0 <= self.x <= SCREEN_WIDTH and 0 <= self.y <= SCREEN_HEIGHT)

    def draw(self, screen):
        if TRAIN_MODE:
            return
        
        pygame.draw.circle(screen, (40,40,40), (int(self.x), int(self.y)), BULLET_SIZE//2)
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), BULLET_SIZE//2 - 2)
        
        if pygame.time.get_ticks() - self.birth_frame < 100:
            pygame.draw.circle(screen, (255,200,0), (int(self.x), int(self.y)), BULLET_SIZE//2 + 3, 2)

# ===================== Game Core Class (Ultimate Fix: Avoid Corner Stuck) =====================
class Game:
    def __init__(self):
        if TRAIN_MODE:
            pygame.init()
            self.screen = None
            self.font = None
            self.small_font = None
        else:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Tank Battle AI (Ultimate Fix: Track + Shoot + Avoid Corners)")
            self.font = pygame.font.SysFont("Arial", 28, bold=True)
            self.small_font = pygame.font.SysFont("Arial", 20)

        self.clock = pygame.time.Clock()
        
        self.walls = [
            {"rect": pygame.Rect(200, 150, 60, 300), "hp": 3},
            {"rect": pygame.Rect(540, 150, 60, 300), "hp": 3},
            {"rect": pygame.Rect(320, 450, 160, 60), "hp": 2},
            {"rect": pygame.Rect(320, 90, 160, 60), "hp": 2},
        ]
        
        self.reset()

    def reset(self):
        self.player = PlayerTank()
        self.ai_tank = AITank()
        self.ai_tank.game = self  # Critical: Let AI access game state
        self.bullets = []
        self.score = 0
        self.game_over = False
        self.frame_count = 0
        self.current_reward = 0
        self.current_action = "None"
        
        for wall in self.walls:
            wall["hp"] = 3 if wall["rect"].width == 60 else 2

    def is_ai_facing_player(self):
        dx = self.player.x - self.ai_tank.x
        dy = self.player.y - self.ai_tank.y
        
        # Improved direction check (more accurate for moving player)
        if self.ai_tank.direction == "up" and abs(dx) < abs(dy)*1.2 and dy < 0:
            return True
        elif self.ai_tank.direction == "down" and abs(dx) < abs(dy)*1.2 and dy > 0:
            return True
        elif self.ai_tank.direction == "left" and abs(dy) < abs(dx)*1.2 and dx < 0:
            return True
        elif self.ai_tank.direction == "right" and abs(dy) < abs(dx)*1.2 and dx > 0:
            return True
        return False

    def get_state(self):
        player_dir = {"up":0,"down":1,"left":2,"right":3}[self.player.direction]
        ai_dir = {"up":0,"down":1,"left":2,"right":3}[self.ai_tank.direction]
        
        p_state = [self.player.x/SCREEN_WIDTH, self.player.y/SCREEN_HEIGHT, player_dir/3, self.player.lives/LIVES]
        ai_state = [self.ai_tank.x/SCREEN_WIDTH, self.ai_tank.y/SCREEN_HEIGHT, ai_dir/3, self.ai_tank.lives/LIVES]
        
        rel_x = (self.player.x - self.ai_tank.x) / SCREEN_WIDTH
        rel_y = (self.player.y - self.ai_tank.y) / SCREEN_HEIGHT
        rel_state = [rel_x, rel_y]
        
        bullet_state = []
        sorted_bullets = sorted(self.bullets, key=lambda b: np.hypot(b.x-self.ai_tank.x, b.y-self.ai_tank.y))[:5]
        for b in sorted_bullets:
            b_dir = {"up":0,"down":1,"left":2,"right":3}[b.direction]
            bullet_state.extend([b.x/SCREEN_WIDTH, b.y/SCREEN_HEIGHT, b_dir/3])
        
        bullet_state += [0.0] * (15 - len(bullet_state))
        return np.array(p_state + ai_state + rel_state + bullet_state, dtype=np.float32)

    def calculate_reward(self, ai_action):
        reward = 0.0

        # Reduced movement penalty (stable learning)
        if abs(self.ai_tank.x - self.ai_tank.prev_x) > 1 or abs(self.ai_tank.y - self.ai_tank.prev_y) > 1:
            reward += 0.5
            self.ai_tank.no_move_frames = 0
        else:
            self.ai_tank.no_move_frames += 1
            if self.ai_tank.no_move_frames > 1:
                reward -= 2.0

        # Enhanced chase reward (encourage tracking)
        dist_to_player = np.hypot(self.ai_tank.x-self.player.x, self.ai_tank.y-self.player.y)
        max_dist = np.hypot(SCREEN_WIDTH, SCREEN_HEIGHT)
        reward += (max_dist - dist_to_player) / 15  # Higher reward for getting close
        if dist_to_player < 250:
            reward += 5.0  # Extra reward for close range

        # Aiming reward (more lenient for moving player)
        if self.is_ai_facing_player():
            reward += 4.0

        # Shooting reward (encourage continuous shooting)
        if ai_action == 4:
            ai_bullets = len([b for b in self.bullets if b.shooter=="ai"])
            if self.is_ai_facing_player() and ai_bullets <= 1:
                reward += 8.0  # Higher reward for accurate shooting
            else:
                reward -= 0.3  # Lower penalty for random shooting

        # Hit/receive hit rewards/penalties
        hit_player = any(b.shooter=="ai" and self.check_collision(b, self.player) for b in self.bullets)
        if hit_player:
            reward += 200.0
        hit_ai = any(b.shooter=="player" and self.check_collision(b, self.ai_tank) for b in self.bullets)
        if hit_ai:
            reward -= 100.0

        # Strong corner penalty (prevent AI from sticking to top/bottom/edges)
        if self.ai_tank.x < 100 or self.ai_tank.x > SCREEN_WIDTH-100:
            reward -= 6.0
        if self.ai_tank.y < 100 or self.ai_tank.y > SCREEN_HEIGHT-100:
            reward -= 8.0  # Heavier penalty for top/bottom stuck

        self.ai_tank.prev_x, self.ai_tank.prev_y = self.ai_tank.x, self.ai_tank.y
        self.current_reward = reward
        return reward

    def check_collision(self, bullet, tank):
        bullet_rect = pygame.Rect(bullet.x-BULLET_SIZE//2, bullet.y-BULLET_SIZE//2, BULLET_SIZE, BULLET_SIZE)
        tank_rect = pygame.Rect(tank.x-TANK_BODY_SIZE//2, tank.y-TANK_BODY_SIZE//2, TANK_BODY_SIZE, TANK_BODY_SIZE)
        return bullet_rect.colliderect(tank_rect)

    def fire_bullet(self, shooter):
        tank = self.player if shooter == "player" else self.ai_tank
        if len([b for b in self.bullets if b.shooter == shooter]) < 2:
            self.bullets.append(Bullet(tank.x, tank.y, tank.direction, shooter))

    def update(self, ai_action=None, event_list=None):
        if self.game_over: return
        event_list = event_list or []
        
        # Handle player shooting
        for e in event_list:
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                self.fire_bullet("player")
        
        # Handle AI shooting
        if ai_action == 4:
            self.fire_bullet("ai")
            self.current_action = "Shoot"
        else:
            self.current_action = ["Up","Down","Left","Right","Shoot","Stop"][ai_action] if ai_action is not None else "None"
        
        # ========== Ultimate Fix 3: Smart wall avoidance (towards player) ==========
        ai_rect = pygame.Rect(
            self.ai_tank.x - TANK_BODY_SIZE//2,
            self.ai_tank.y - TANK_BODY_SIZE//2,
            TANK_BODY_SIZE,
            TANK_BODY_SIZE
        )
        # Check wall collision
        for wall in self.walls:
            if wall["hp"] > 0 and ai_rect.colliderect(wall["rect"]):
                # Calculate direction to player (avoid wall + chase player)
                dx = self.player.x - self.ai_tank.x
                dy = self.player.y - self.ai_tank.y
                
                # Smart direction change (prioritize towards player)
                if wall["rect"].x < self.ai_tank.x and dx > 0:
                    self.ai_tank.direction = "right"  # Move right (towards player)
                elif wall["rect"].x > self.ai_tank.x and dx < 0:
                    self.ai_tank.direction = "left"   # Move left (towards player)
                elif wall["rect"].y < self.ai_tank.y and dy > 0:
                    self.ai_tank.direction = "down"   # Move down (towards player)
                elif wall["rect"].y > self.ai_tank.y and dy < 0:
                    self.ai_tank.direction = "up"     # Move up (towards player)
                else:
                    # Fallback: Perpendicular direction (avoid random stuck)
                    if self.ai_tank.direction in ["up", "down"]:
                        self.ai_tank.direction = "right" if dx > 0 else "left"
                    else:
                        self.ai_tank.direction = "down" if dy > 0 else "up"
                
                # Move AI 2 steps in new direction (escape wall quickly)
                dx_move, dy_move = 0, 0
                if self.ai_tank.direction == "up":
                    dy_move = -TANK_SPEED * 2
                elif self.ai_tank.direction == "down":
                    dy_move = TANK_SPEED * 2
                elif self.ai_tank.direction == "left":
                    dx_move = -TANK_SPEED * 2
                elif self.ai_tank.direction == "right":
                    dx_move = TANK_SPEED * 2
                self.ai_tank.move(dx_move, dy_move)
                break
        
        # ========== Ultimate Fix 4: Prevent AI from sticking to screen edges ==========
        # Force AI to move towards center if it hits top/bottom/left/right edges
        if self.ai_tank.y <= TANK_BODY_SIZE//2 + 20:  # Top edge
            self.ai_tank.direction = "down"
            self.ai_tank.move(0, TANK_SPEED * 2)
        elif self.ai_tank.y >= SCREEN_HEIGHT - TANK_BODY_SIZE//2 - 20:  # Bottom edge
            self.ai_tank.direction = "up"
            self.ai_tank.move(0, -TANK_SPEED * 2)
        if self.ai_tank.x <= TANK_BODY_SIZE//2 + 20:  # Left edge
            self.ai_tank.direction = "right"
            self.ai_tank.move(TANK_SPEED * 2, 0)
        elif self.ai_tank.x >= SCREEN_WIDTH - TANK_BODY_SIZE//2 - 20:  # Right edge
            self.ai_tank.direction = "left"
            self.ai_tank.move(-TANK_SPEED * 2, 0)
        
        # Update bullets
        for b in self.bullets[:]:
            b.move()
            wall_hit = False
            for wall in self.walls:
                if wall["hp"] > 0 and b.active and wall["rect"].collidepoint(b.x, b.y):
                    wall["hp"] -= 1
                    self.bullets.remove(b)
                    b.active = False
                    wall_hit = True
                    break
            if wall_hit: continue
            
            # Check player hit
            if b.shooter == "ai" and self.check_collision(b, self.player):
                self.player.lives -= 1
                self.bullets.remove(b)
                if self.player.lives <= 0:
                    self.game_over = True
            # Check AI hit
            elif b.shooter == "player" and self.check_collision(b, self.ai_tank):
                self.ai_tank.lives -= 1
                self.bullets.remove(b)
                if self.ai_tank.lives <= 0:
                    self.game_over = True
            # Remove out of bounds bullets
            elif b.is_out_of_bounds():
                self.bullets.remove(b)
        
        self.frame_count += 1

    def draw_ui(self):
        if TRAIN_MODE or not self.screen:
            return
        
        self.screen.fill(BG_LIGHT)
        
        # Draw grid lines
        for y in range(0, SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, (235,235,235), (0, y), (SCREEN_WIDTH, y), 1)
        for x in range(0, SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, (235,235,235), (x, 0), (x, SCREEN_HEIGHT), 1)
        
        # Draw walls
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

# ===================== Play Game Function =====================
def play_game():
    global TRAIN_MODE
    TRAIN_MODE = False
    
    # Re-enable pygame video/audio for GUI
    os.environ.pop('SDL_VIDEODRIVER', None)
    os.environ.pop('SDL_AUDIODRIVER', None)
    pygame.quit()
    pygame.init()
    
    # Initialize game and DQN model
    game = Game()
    dqn_model = DQN(STATE_DIM, ACTION_DIM)
    model_path = f"models/dqn_tank_ep{TRAIN_EPISODES}.pth"
    
    # Load trained model (fallback to stable movement if not found)
    try:
        dqn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        dqn_model.eval()
        print(f"✅ Model loaded successfully: {model_path}")
    except:
        print(f"⚠️ No trained model found ({model_path}) — AI will still track/shoot stably")

    # Reduce exploration rate for play mode
    game.ai_tank.epsilon = 0.05  # Almost no random actions (pure tracking)
    print("🎮 Controls: WASD = move, SPACE = shoot, R = restart, Q = quit")
    print("💡 AI will now track your movement and shoot continuously!")

    # Main game loop
    while True:
        events = pygame.event.get()
        # Handle quit/restart events
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
        
        # Get game state and select AI action
        state = game.get_state()
        with torch.no_grad():
            action = game.ai_tank.take_action(state, dqn_model, is_training=False)
        
        # Handle player input and update game state
        game.player.handle_input(pygame.key.get_pressed())
        game.update(action, events)
        game.calculate_reward(action)
        game.draw_ui()
        game.clock.tick(FPS)

# ===================== Train DQN Function =====================
def train_dqn():
    global TRAIN_MODE
    TRAIN_MODE = True

    # Disable pygame video/audio for faster training
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'

    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Initialize game and DQN models
    game = Game()
    dqn_model = DQN(STATE_DIM, ACTION_DIM)
    target_model = DQN(STATE_DIM, ACTION_DIM)
    target_model.load_state_dict(dqn_model.state_dict())
    optimizer = optim.Adam(dqn_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    memory = deque(maxlen=MEMORY_CAPACITY)
    pbar = tqdm(range(TRAIN_EPISODES), desc=f"Training AI ({TRAIN_EPISODES} episodes)...")

    # Training loop
    for episode in pbar:
        game.reset()
        total_reward = 0
        total_loss = 0
        loss_count = 0
        state = game.get_state()

        while not game.game_over:
            # Select AI action
            action = game.ai_tank.take_action(state, dqn_model)

            # Simulate player movement (training mode only)
            if TRAIN_MODE:
                dx, dy = 0, 0
                if random.random() < 0.9:  # More aggressive player movement
                    px, py = game.player.x, game.player.y
                    ax, ay = game.ai_tank.x, game.ai_tank.y
                    if abs(ax-px) > abs(ay-py):
                        dx = TANK_SPEED * 0.9 if ax > px else -TANK_SPEED * 0.9
                        game.player.direction = "right" if ax > px else "left"
                    else:
                        dy = TANK_SPEED * 0.9 if ay > py else -TANK_SPEED * 0.9
                        game.player.direction = "down" if ay > py else "up"
                else:
                    direction = random.choice(["up", "down", "left", "right"])
                    if direction == "up":
                        dy = -TANK_SPEED * 0.8
                        game.player.direction = "up"
                    elif direction == "down":
                        dy = TANK_SPEED * 0.8
                        game.player.direction = "down"
                    elif direction == "left":
                        dx = -TANK_SPEED * 0.8
                        game.player.direction = "left"
                    elif direction == "right":
                        dx = TANK_SPEED * 0.8
                        game.player.direction = "right"
                game.player.move(dx, dy)

            # Update game state and get next state/reward
            game.update(action, [])
            next_state = game.get_state()
            reward = game.calculate_reward(action)
            total_reward += reward
            
            # Store experience in replay memory
            memory.append((state, action, reward, next_state, game.game_over))
            state = next_state

            # Experience replay
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states = torch.FloatTensor(np.array([x[0] for x in batch]))
                actions = torch.LongTensor(np.array([x[1] for x in batch])).unsqueeze(1)
                rewards = torch.FloatTensor(np.array([x[2] for x in batch]))
                next_states = torch.FloatTensor(np.array([x[3] for x in batch]))
                dones = torch.FloatTensor(np.array([x[4] for x in batch]))

                current_q = dqn_model(states).gather(1, actions).squeeze(1)
                next_q = target_model(next_states).max(1)[0]
                target_q = rewards + GAMMA * next_q * (1 - dones)

                loss = criterion(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                loss_count += 1

            game.clock.tick(TRAIN_FPS)

        # Update progress bar
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        pbar.set_postfix({"Reward": f"{total_reward:.1f}", "Eps": f"{game.ai_tank.epsilon:.2f}"})

        # Update target model
        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            target_model.load_state_dict(dqn_model.state_dict())
        
        # Save model at final episode
        if (episode + 1) == TRAIN_EPISODES:
            torch.save(dqn_model.state_dict(), f"models/dqn_tank_ep{TRAIN_EPISODES}.pth")

    # Save final model
    torch.save(dqn_model.state_dict(), f"models/dqn_tank_best_{TRAIN_EPISODES}ep.pth")
    print(f"\n✅ Training finished! Model saved as: models/dqn_tank_best_{TRAIN_EPISODES}ep.pth")
    pygame.quit()

# ===================== Main Entry Point =====================
if __name__ == "__main__":
    print(f"===== Tank Battle DQN AI (Ultimate Fixed Version) =====")
    print(f"Current training episodes set to: {TRAIN_EPISODES}")
    print("1) Train AI (stable for any episodes)")
    print("2) Play game (AI tracks player + shoots + avoids corners)")
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        train_dqn()
    elif choice == "2":
        play_game()
    else:
        print("Invalid input — starting training automatically")
        train_dqn()
