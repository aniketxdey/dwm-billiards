#!/usr/bin/env python3
"""
Neural Newton - Billiard/Pool Style Recording
Top-down pool table with pockets.
"""

import random
import math
import numpy as np
import pymunk
import cv2
import config


# Authentic billiard table colors
FELT_COLOR = (39, 119, 78)  # Classic billiard green
CUSHION_COLOR = (101, 67, 33)  # Rich mahogany wood
POCKET_COLOR = (20, 20, 20)  # Deep black pockets
RAIL_HIGHLIGHT = (130, 90, 50)  # Wood highlight

# Real pool ball colors (solids 1-7, black 8, stripes simulated as lighter)
BALL_COLORS = [
    (255, 255, 255),  # Cue ball - white
    (255, 215, 0),    # 1 - Yellow
    (0, 0, 180),      # 2 - Blue
    (255, 50, 50),    # 3 - Red
    (128, 0, 128),    # 4 - Purple/Plum
    (255, 120, 0),    # 5 - Orange
    (34, 139, 34),    # 6 - Green
    (139, 69, 19),    # 7 - Maroon/Brown
    (20, 20, 20),     # 8 - Black
    (255, 230, 100),  # 9 - Yellow stripe (lighter)
    (100, 100, 220),  # 10 - Blue stripe
    (255, 130, 130),  # 11 - Red stripe
    (180, 100, 180),  # 12 - Purple stripe
    (255, 180, 100),  # 13 - Orange stripe
    (100, 180, 100),  # 14 - Green stripe
    (180, 120, 80),   # 15 - Maroon stripe
]

# Pocket positions (corners only for simplicity)
POCKET_RADIUS = 6
def get_pockets():
    w, h = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
    return [
        (5, 5),           # Top-left
        (w - 5, 5),       # Top-right
        (5, h - 5),       # Bottom-left
        (w - 5, h - 5),   # Bottom-right
    ]


class BilliardWorld:
    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # Top-down, no gravity
        self.space.damping = config.DAMPING  # Balls slow down
        
        self.balls = []
        self.grabbed = None
        self.pockets = get_pockets()
        self.pocketed = 0  # Score counter
        
        self._walls()
        self._spawn()
    
    def _walls(self):
        # Cushions (walls)
        t = 3
        w, h = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
        walls = [
            ((t, t), (w - t, t)),         # Top
            ((t, h - t), (w - t, h - t)), # Bottom
            ((t, t), (t, h - t)),         # Left
            ((w - t, t), (w - t, h - t)), # Right
        ]
        for start, end in walls:
            wall = pymunk.Segment(self.space.static_body, start, end, t)
            wall.elasticity = config.RESTITUTION
            wall.friction = 0.5
            self.space.add(wall)
    
    def _spawn(self):
        n = random.randint(config.BALL_COUNT_MIN, config.BALL_COUNT_MAX)
        w, h = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
        r = config.BALL_RADIUS  # Uniform size
        
        placed = []  # Track placed ball positions
        
        for i in range(n):
            m = config.BALL_MASS
            
            # Try to find non-overlapping position
            margin = r + 10
            for attempt in range(100):
                x = random.uniform(margin, w - margin)
                y = random.uniform(margin, h - margin)
                
                # Check distance from all placed balls
                valid = True
                for px, py in placed:
                    dist = math.sqrt((x - px)**2 + (y - py)**2)
                    if dist < r * 2.5:  # Need space between balls
                        valid = False
                        break
                
                if valid:
                    break
            
            placed.append((x, y))
            
            body = pymunk.Body(m, pymunk.moment_for_circle(m, 0, r))
            body.position = (x, y)
            body.velocity = (0, 0)  # Start stationary
            
            shape = pymunk.Circle(body, r)
            shape.elasticity = config.RESTITUTION
            shape.friction = config.FRICTION
            
            self.space.add(body, shape)
            self.balls.append((body, r, BALL_COLORS[i % len(BALL_COLORS)]))
    
    def grab(self, idx):
        if 0 <= idx < len(self.balls):
            self.grabbed = self.balls[idx][0]
    
    def release(self):
        if self.grabbed:
            # Give a little push when releasing
            self.grabbed = None
    
    def flick(self, velocity):
        """Flick the grabbed ball with velocity."""
        if self.grabbed:
            self.grabbed.velocity = velocity
            self.grabbed = None
    
    def move_grabbed(self, pos):
        if self.grabbed:
            x = max(8, min(config.SCREEN_WIDTH - 8, pos[0]))
            y = max(8, min(config.SCREEN_HEIGHT - 8, pos[1]))
            self.grabbed.position = (x, y)
            self.grabbed.velocity = (0, 0)
    
    def step(self, dt):
        self.space.step(dt)
        
        # Check for balls in pockets
        to_remove = []
        for i, (body, r, color) in enumerate(self.balls):
            for px, py in self.pockets:
                dist = math.sqrt((body.position.x - px)**2 + (body.position.y - py)**2)
                if dist < POCKET_RADIUS:
                    to_remove.append(i)
                    break
        
        # Remove pocketed balls (reverse order to maintain indices)
        for i in reversed(to_remove):
            body, r, color = self.balls[i]
            for shape in body.shapes:
                self.space.remove(shape)
            self.space.remove(body)
            self.balls.pop(i)
            self.pocketed += 1
    
    def render(self, scale):
        w, h = config.SCREEN_WIDTH * scale, config.SCREEN_HEIGHT * scale
        
        # Green felt background
        img = np.full((h, w, 3), FELT_COLOR[::-1], dtype=np.uint8)
        
        # Draw wood rail (outer)
        rail = 5 * scale
        cv2.rectangle(img, (0, 0), (w-1, h-1), CUSHION_COLOR[::-1], rail)
        # Wood highlight on top edge
        cv2.rectangle(img, (0, 0), (w-1, rail//2), RAIL_HIGHLIGHT[::-1], -1)
        # Inner cushion (darker green edge)
        cushion = 2 * scale
        cv2.rectangle(img, (rail-cushion, rail-cushion), 
                     (w-rail+cushion, h-rail+cushion), (30, 100, 60), cushion)
        
        # Draw pockets
        for px, py in self.pockets:
            cx, cy = int(px * scale), int(py * scale)
            pr = int(POCKET_RADIUS * scale)
            cv2.circle(img, (cx, cy), pr, POCKET_COLOR[::-1], -1, cv2.LINE_AA)
        
        # Draw balls - solid colors with dark border
        for body, radius, color in self.balls:
            x = int(body.position.x * scale)
            y = int(body.position.y * scale)
            r = int(radius * scale)
            # Dark border
            cv2.circle(img, (x, y), r + 1, (20, 20, 20), -1, cv2.LINE_AA)
            # Solid ball color (no highlight)
            cv2.circle(img, (x, y), r, color[::-1], -1, cv2.LINE_AA)
        
        # Simple score in corner (small, unobtrusive)
        score_text = str(self.pocketed)
        cv2.putText(img, score_text, (w - 30, 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        return img


class PoolBot:
    """
    Bot that shoots balls with varied strategies.
    Each bot has a randomized "personality" for realistic variety:
    - Strategy weights (random/ball/pocket/bank/tap)
    - Speed multiplier
    - Accuracy (error std)
    """
    
    def __init__(self, world):
        self.world = world
        self.pos = (64, 36)
        self.state = "pick"
        self.timer = 0.2
        self.ball_idx = -1
        self.shoot_dir = (0, 0)
        self.shot_power = 100
        
        # Randomize bot personality
        self._randomize_personality()
    
    def _randomize_personality(self):
        """Create a unique bot personality for this episode."""
        # Strategy weights (will be normalized)
        # Pocket weight is dominant to ensure lots of scoring attempts
        self.w_random = random.uniform(0.02, 0.08)
        self.w_ball = random.uniform(0.05, 0.15)
        self.w_pocket = random.uniform(0.60, 0.80)  # Extreme focus on scoring
        self.w_bank = random.uniform(0.01, 0.05)
        self.w_tap = random.uniform(0.01, 0.05)
        
        # Normalize to sum to 1
        total = self.w_random + self.w_ball + self.w_pocket + self.w_bank + self.w_tap
        self.w_random /= total
        self.w_ball /= total
        self.w_pocket /= total
        self.w_bank /= total
        self.w_tap /= total
        
        # Speed multiplier (0.8 = slower, 1.5 = faster)
        self.speed_mult = random.uniform(0.9, 1.4)
        
        # Accuracy (lower = more accurate)
        self.accuracy = random.uniform(0.02, 0.08)
    
    def _pick_shoot_direction(self):
        """
        Pick shoot direction and power based on strategy.
        Returns: ((dx, dy), power)
        """
        ball = self.world.balls[self.ball_idx][0]
        bx, by = ball.position.x, ball.position.y
        
        roll = random.random()
        
        # Helper for adding human error
        def humanize(dx, dy, error_std=None):
            if error_std is None:
                error_std = self.accuracy
            angle = math.atan2(dy, dx)
            angle += random.gauss(0, error_std)
            return (math.cos(angle), math.sin(angle))
        
        # Cumulative thresholds based on personality
        t1 = self.w_random
        t2 = t1 + self.w_ball
        t3 = t2 + self.w_pocket
        t4 = t3 + self.w_bank
        # t5 = 1.0 (tap)
            
        if roll < t1:
            # Random direction
            angle = random.uniform(0, 2 * math.pi)
            power = random.uniform(60, 140)
            return ((math.cos(angle), math.sin(angle)), power)
        
        elif roll < t2:
            # Aim at another ball
            other_balls = [i for i in range(len(self.world.balls)) if i != self.ball_idx]
            if other_balls:
                target_idx = random.choice(other_balls)
                target = self.world.balls[target_idx][0]
                dx = target.position.x - bx
                dy = target.position.y - by
                dist = math.sqrt(dx*dx + dy*dy)
                
                # Power relative to distance + kick
                base_power = min(150, max(50, dist * 1.5))
                power = base_power * random.uniform(0.8, 1.2)
                
                if dist > 0.1:
                    return (humanize(dx, dy), power)
            
            # Fallback to random
            angle = random.uniform(0, 2 * math.pi)
            return ((math.cos(angle), math.sin(angle)), random.uniform(60, 140))
            
        elif roll < t3:
            # Aim at pocket (try to score) - pick NEAREST pocket for higher success
            w, h = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
            pockets = [(5, 5), (w-5, 5), (5, h-5), (w-5, h-5)]
            
            # 90% chance to pick nearest pocket, 10% random (was 70/30)
            if random.random() < 0.9:
                # Find nearest pocket
                best_pocket = None
                best_dist = float('inf')
                for p in pockets:
                    d = math.sqrt((p[0] - bx)**2 + (p[1] - by)**2)
                    if d < best_dist:
                        best_dist = d
                        best_pocket = p
                target = best_pocket
            else:
                target = random.choice(pockets)
            
            dx = target[0] - bx
            dy = target[1] - by
            dist = math.sqrt(dx*dx + dy*dy)
            
            # Power calibrated for distance - not too hard, not too soft
            # Closer = softer shot, farther = harder
            base_power = min(140, max(60, dist * 1.5))
            power = base_power * random.uniform(0.95, 1.05)  # Less variance for accuracy
            
            if dist > 0.1:
                # Almost perfect aim for pocket shots (sniper mode)
                return (humanize(dx, dy, 0.01), power)
                
            angle = random.uniform(0, 2 * math.pi)
            return ((math.cos(angle), math.sin(angle)), random.uniform(60, 140))
        
        elif roll < t4:
            # Smart bank shot (aim at wall to hit another ball)
            other_balls = [i for i in range(len(self.world.balls)) if i != self.ball_idx]
            if other_balls:
                target_idx = random.choice(other_balls)
                target_ball = self.world.balls[target_idx][0]
                tx, ty = target_ball.position.x, target_ball.position.y
                w, h = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
                
                # Wall offset (cushion thickness + ball radius)
                offset = 5
                
                # Pick a random wall to bounce off
                wall = random.choice(['top', 'bottom', 'left', 'right'])
                
                if wall == 'top':
                    # Reflect target across y=offset
                    aim_x, aim_y = tx, 2*offset - ty
                elif wall == 'bottom':
                    # Reflect across y=(h-offset)
                    aim_x, aim_y = tx, 2*(h - offset) - ty
                elif wall == 'left':
                    # Reflect across x=offset
                    aim_x, aim_y = 2*offset - tx, ty
                else:  # right
                    # Reflect across x=(w-offset)
                    aim_x, aim_y = 2*(w - offset) - tx, ty
                
                dx = aim_x - bx
                dy = aim_y - by
                dist = math.sqrt(dx*dx + dy*dy)
                
                # Bank shots need more power to travel further
                power = min(180, max(80, dist * 1.5)) * random.uniform(0.9, 1.2)
                
                if dist > 0.1:
                    # More error on bank shots (harder to aim)
                    return (humanize(dx, dy, self.accuracy * 1.5), power)
            
            # Fallback
            angle = random.uniform(0, 2 * math.pi)
            return ((math.cos(angle), math.sin(angle)), random.uniform(60, 140))
        
        else:
            # Gentle tap - 10%
            angle = random.uniform(0, 2 * math.pi)
            power = random.uniform(20, 50)  # Very weak
            return ((math.cos(angle), math.sin(angle)), power)
    
    def do_action(self):
        """Trigger a single bot action (for interactive testing)."""
        if len(self.world.balls) == 0:
            return
        self.ball_idx = random.randint(0, len(self.world.balls) - 1)
        self.state = "goto"
        self.timer = 1.0
    
    def update(self, dt):
        self.timer -= dt
        
        if len(self.world.balls) == 0:
            return
        
        if self.state == "pick":
            if self.timer <= 0:
                self.ball_idx = random.randint(0, len(self.world.balls) - 1)
                self.state = "goto"
                self.timer = 0.6 / self.speed_mult
        
        elif self.state == "goto":
            if self.ball_idx >= len(self.world.balls):
                self.state = "pick"
                self.timer = 0.1
                return
            
            ball = self.world.balls[self.ball_idx][0]
            tx, ty = ball.position.x, ball.position.y
            dx, dy = tx - self.pos[0], ty - self.pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > 2:
                speed = 90 * self.speed_mult
                self.pos = (self.pos[0] + dx/dist * speed * dt,
                           self.pos[1] + dy/dist * speed * dt)
            else:
                # Pick strategy: Direction AND Power
                self.shoot_dir, self.shot_power = self._pick_shoot_direction()
                
                self.world.grab(self.ball_idx)
                self.state = "shoot"
                self.timer = 0.10 / self.speed_mult
            
            if self.timer <= 0:
                self.state = "pick"
                self.timer = 0.1
        
        elif self.state == "shoot":
            drag_speed = 40 * self.speed_mult
            self.pos = (self.pos[0] + self.shoot_dir[0] * drag_speed * dt,
                       self.pos[1] + self.shoot_dir[1] * drag_speed * dt)
            self.world.move_grabbed(self.pos)
            
            if self.timer <= 0:
                if self.world.grabbed:
                    self.world.flick((-self.shoot_dir[0] * self.shot_power, 
                                     -self.shoot_dir[1] * self.shot_power))
                self.state = "watch"
                # Watch longer if power was high
                base_watch = (0.2 + (self.shot_power / 200.0) * 0.4) / self.speed_mult
                self.timer = random.uniform(base_watch, base_watch + 0.3)
        
        elif self.state == "watch":
            if self.timer <= 0:
                self.state = "pick"
                self.timer = random.uniform(0.01, 0.08) / self.speed_mult


def main():
    print("Recording billiard demo...")
    
    world = BilliardWorld()
    bot = PoolBot(world)
    
    scale = 8
    w, h = config.SCREEN_WIDTH * scale, config.SCREEN_HEIGHT * scale
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('billiard_demo.avi', fourcc, config.FPS, (w, h))
    
    dt = 1.0 / config.FPS
    frames = 20 * config.FPS  # 20 seconds
    
    for i in range(frames):
        bot.update(dt)
        world.step(dt)
        frame = world.render(scale)
        out.write(frame)
        
        if i % 100 == 0:
            print(f"  {i}/{frames}")
    
    out.release()
    print("Done! Saved to billiard_demo.avi")
    
    # Convert to MP4
    import subprocess
    subprocess.run([
        'ffmpeg', '-y', '-i', 'billiard_demo.avi',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '22',
        'billiard_demo.mp4'
    ], capture_output=True)
    print("Converted to billiard_demo.mp4")


if __name__ == "__main__":
    main()
