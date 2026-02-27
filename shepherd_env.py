import gym
from gym import spaces
import numpy as np
import pygame

class Shepherd: # Gardé pour compatibilité si utilisé ailleurs
    def __init__(self, pos, wheel_base=15.0):
        self.pos = np.array(pos, dtype=float)
        self.theta = np.random.uniform(0, 2*np.pi)
        self.wheel_base = wheel_base

class Sheep:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.is_safe = False

class ShepherdEnv(gym.Env):
    def __init__(
        self,
        n_sheep=1,
        world_size=1.0,
        goal_radius=0.7,
        obstacle_radius=0,
        sheep_repulsion_radius=0.2,
        shepherd_speed=0.05,
        max_steps=500,
        wandering_strength=0.0  # <--- NOUVEAU : Pour le Level 2
    ):
        super().__init__()

        self.n_sheep = n_sheep
        self.world_size = world_size
        self.goal_radius = goal_radius
        
        if obstacle_radius > 0.3:
            print("Warning: obstacle_radius too large, setting to 0.3")
            self.obstacle_radius = 0.3
        else:
            self.obstacle_radius = obstacle_radius
            
        self.repulsion_radius = sheep_repulsion_radius
        self.shepherd_speed = shepherd_speed
        self.max_steps = max_steps
        self.wandering_strength = wandering_strength # <--- NOUVEAU

        # Action: shepherd orientation angle in degrees [-180, +180]
        # On normalise entre -1 et 1 pour le réseau de neurones
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

        # Observation:
        # Pour chaque mouton: [sheep_rel_x, sheep_rel_y, goal_rel_x, goal_rel_y] (4)
        # Goal relatif au berger (2)
        # Obstacle relatif au berger (2)
        obs_dim = 4 * self.n_sheep + 2 + 2
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.steps = 0
        # Positions aléatoires
        self.shepherd = np.random.uniform(-0.8, 0.8, size=2)
        self.obstacle = np.random.uniform(-0.8, 0.8, size=2)
        self.goal = np.random.uniform(-0.8, 0.8, size=2)
        
        # S'assurer que l'obstacle n'est pas sur le goal
        while np.linalg.norm(self.obstacle - self.goal) < (self.goal_radius + self.obstacle_radius + 0.1):
            self.goal = np.random.uniform(-0.8, 0.8, size=2)
            self.obstacle = np.random.uniform(-0.8, 0.8, size=2)

        self.sheep = [
            np.random.uniform(-0.8, 0.8, size=2)
            for _ in range(self.n_sheep)
        ]
        # S'assurer que les moutons ne commencent pas dans le goal
        for i, s in enumerate(self.sheep):
            while np.linalg.norm(s - self.goal) < self.goal_radius:
                s = np.random.uniform(-0.8, 0.8, size=2)
                self.sheep[i] = s

        self.prev_goal_dist = self._mean_sheep_goal_dist()
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for s in self.sheep:
            sheep_rel = s - self.shepherd
            goal_rel_s = self.goal - s
            obs.extend(sheep_rel)
            obs.extend(goal_rel_s)

        goal_rel = self.goal - self.shepherd
        obs.extend(goal_rel)
        
        # Ajout obstacle seulement si radius > 0 (Level 3), sinon on met des 0 pour ne pas casser le modèle
        if self.obstacle_radius > 0:
            obstacle_rel = self.obstacle - self.shepherd
        else:
            obstacle_rel = np.zeros(2)
            
        obs.extend(obstacle_rel)

        return np.clip(np.array(obs, dtype=np.float32), -1.0, 1.0)

    def _mean_sheep_goal_dist(self):
        return np.mean([np.linalg.norm(s - self.goal) for s in self.sheep])

    def _max_sheep_goal_dist(self):
        return np.max([np.linalg.norm(s - self.goal) for s in self.sheep])

    def step(self, action):
        self.steps += 1
        self.prev_shepherd = self.shepherd.copy()

        # --- Shepherd dynamics ---
        angle_deg = float(np.clip(action[0]*180, -180.0, 180.0))
        angle_rad = np.deg2rad(angle_deg)

        move = np.array([
            np.cos(angle_rad),
            np.sin(angle_rad)
        ]) * self.shepherd_speed

        self.shepherd += move
        self.shepherd = np.clip(self.shepherd, -1.0, 1.0)

        # --- Sheep dynamics ---
        for i, s in enumerate(self.sheep):
            move = np.zeros(2)

            # 1. Répulsion berger
            vec = s - self.shepherd
            dist = np.linalg.norm(vec)

            if dist < self.repulsion_radius:
                move += (vec / (dist + 1e-6)) * 0.05

            # 2. Level 2: Wandering (Bruit aléatoire) <--- NOUVEAU
            if self.wandering_strength > 0:
                move += np.random.normal(0, self.wandering_strength, size=2)

            # Mise à jour position avec gestion obstacle
            next_pos = s + move
            if self.obstacle_radius > 0:
                # Si le mouvement fait rentrer le mouton dans l'obstacle, on annule
                if np.linalg.norm(np.clip(next_pos, -0.9, 0.9) - self.obstacle) < (self.obstacle_radius + 0.05):
                    # Rebond simple ou arrêt
                    pass 
                else:
                    self.sheep[i] = np.clip(next_pos, -0.9, 0.9)
            else:
                self.sheep[i] = np.clip(next_pos, -0.9, 0.9)

        # --- Reward ---
        reward = 0.0

        # 1. Progress
        curr_dist = self._mean_sheep_goal_dist()
        reward += (self.prev_goal_dist - curr_dist) * 300.0
        self.prev_goal_dist = curr_dist

        # 2. Proximity target
        sheep_dists = np.linalg.norm(np.array(self.sheep) - self.shepherd, axis=1)
        furthest_idx = np.argmax([np.linalg.norm(s - self.goal) for s in self.sheep])
        dist_to_target_sheep = sheep_dists[furthest_idx]
        reward += 5.0 * np.exp(-5.0 * dist_to_target_sheep)

        # 3. Regularization
        shepherd_move = np.linalg.norm(self.shepherd - self.prev_shepherd)
        reward -= 10 * np.exp(-100.0 * shepherd_move)

        # --- Termination ---
        done = False
        if self._max_sheep_goal_dist() < self.goal_radius:
            reward += 200.0 * self.n_sheep
            reward += 5 * (self.max_steps - self.steps)
            done = True
        else:
            for s in self.sheep:
                if np.linalg.norm(s - self.goal) < self.goal_radius:
                    reward += 100.0

        if self.steps >= self.max_steps:
            done = True
            reward -= 10.0
        else:
            reward -= 0.02

        return self._get_obs(), reward, done, {}
    
    def render(self, mode='human'):
        if not hasattr(self, "screen") or self.screen is None:
            pygame.init()
            self.screen_size_px = 600
            self.screen = pygame.display.set_mode((self.screen_size_px, self.screen_size_px))
            pygame.display.set_caption("Shepherd Environment")
            self.clock = pygame.Clock()
            self.font = pygame.font.Font(None, 24)

        self.screen.fill((255, 255, 255))

        def to_px(pos):
            return ((pos + 1) * self.screen_size_px / 2).astype(int)

        # Draw goal
        pygame.draw.circle(
            self.screen, (200, 0, 0),
            to_px(self.goal), int(self.goal_radius * self.screen_size_px/2), width=8
        )

        # Draw obstacle
        if self.obstacle_radius > 0:
            pygame.draw.circle(
                self.screen, (100, 100, 100),
                to_px(self.obstacle), int(self.obstacle_radius * self.screen_size_px / 2), width=0
            )

        # Draw sheep
        for s in self.sheep:
            pygame.draw.circle(
                self.screen, (0, 0, 0),
                to_px(s), int(0.02 * self.screen_size_px), width=0
            )

        # Draw shepherd
        pygame.draw.circle(
            self.screen, (200, 0, 0),
            to_px(self.shepherd), int(0.03 * self.screen_size_px), width=0
        )

        text = f"Step: {self.steps}/{self.max_steps}"
        text_surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)