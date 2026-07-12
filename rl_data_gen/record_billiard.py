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
# Dark navy pockets — distinct from the black 8-ball (20,20,20) so all 16 balls
# are detectable in physics-grounded evaluation (AAAI-27 benchmark).
POCKET_COLOR = (12, 28, 68)
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
        # v3: explicit episode profile so we can bias toward contact-heavy layouts while
        # still keeping a slice of more regular gameplay trajectories.
        self.event_profile = random.choices(
            ["collision_heavy", "mixed_random"],
            weights=[0.75, 0.25],
            k=1,
        )[0]

        if self.event_profile == "collision_heavy":
            table_weights = [0.25, 0.32, 0.08, 0.35]  # standard, slick, grippy, bouncy
            spawn_weights = [0.06, 0.26, 0.22, 0.12, 0.34]  # uniform, clustered, two_cluster, rail_bias, break_setup
            self.collision_bias_strength = random.uniform(0.75, 1.0)
        else:
            table_weights = [0.40, 0.20, 0.18, 0.22]
            spawn_weights = [0.18, 0.24, 0.18, 0.18, 0.22]
            self.collision_bias_strength = random.uniform(0.35, 0.70)

        self.table_profile = random.choices(
            ["standard", "slick", "grippy", "bouncy"],
            weights=table_weights,
            k=1,
        )[0]
        profile_params = {
            "standard": {"damping": (0.975, 0.985), "restitution": (0.86, 0.92), "friction": (0.88, 0.98)},
            "slick": {"damping": (0.985, 0.993), "restitution": (0.88, 0.95), "friction": (0.82, 0.92)},
            "grippy": {"damping": (0.955, 0.975), "restitution": (0.82, 0.90), "friction": (0.95, 1.05)},
            "bouncy": {"damping": (0.970, 0.982), "restitution": (0.92, 0.98), "friction": (0.86, 0.96)},
        }[self.table_profile]
        self.damping = random.uniform(*profile_params["damping"])
        self.restitution = random.uniform(*profile_params["restitution"])
        self.ball_friction = random.uniform(*profile_params["friction"])
        self.space.damping = self.damping

        self.balls = []
        self.grabbed = None
        self.pockets = get_pockets()
        self.pocketed = 0  # Score counter
        self.spawn_mode = random.choices(
            ["uniform", "clustered", "two_cluster", "rail_bias", "break_setup"],
            weights=spawn_weights,
            k=1,
        )[0]

        self._walls()
        self._spawn()

    def _cluster_spread_scale(self) -> float:
        if self.event_profile == "collision_heavy":
            return random.uniform(0.65, 0.90)
        return random.uniform(0.90, 1.15)

    def _min_spawn_separation(self, r: float) -> float:
        if self.spawn_mode in ("clustered", "two_cluster", "break_setup"):
            if self.event_profile == "collision_heavy":
                return r * random.uniform(1.95, 2.12)
            return r * 2.2
        if self.event_profile == "collision_heavy":
            return r * random.uniform(2.15, 2.35)
        return r * 2.5
    
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
            wall.elasticity = self.restitution
            wall.friction = 0.45
            self.space.add(wall)

    def _sample_spawn_position(self, margin, w, h, r, placed):
        mode = self.spawn_mode
        if mode == "break_setup":
            mode = "clustered"
        spread_scale = self._cluster_spread_scale()
        min_sep = self._min_spawn_separation(r)

        for _ in range(100):
            if mode == "uniform":
                x = random.uniform(margin, w - margin)
                y = random.uniform(margin, h - margin)
            elif mode == "clustered":
                cx = random.uniform(w * 0.35, w * 0.65)
                cy = random.uniform(h * 0.30, h * 0.70)
                x = random.gauss(cx, 7.5 * spread_scale)
                y = random.gauss(cy, 5.5 * spread_scale)
            elif mode == "two_cluster":
                cxs = [w * 0.33, w * 0.67]
                cx = random.choice(cxs) + random.uniform(-4, 4)
                cy = random.uniform(h * 0.25, h * 0.75)
                x = random.gauss(cx, 5.5 * spread_scale)
                y = random.gauss(cy, 4.5 * spread_scale)
            elif mode == "rail_bias":
                if random.random() < 0.5:
                    x = random.uniform(margin, w - margin)
                    y = random.choice([random.uniform(margin, h * 0.22), random.uniform(h * 0.78, h - margin)])
                else:
                    x = random.choice([random.uniform(margin, w * 0.22), random.uniform(w * 0.78, w - margin)])
                    y = random.uniform(margin, h - margin)
            else:
                x = random.uniform(margin, w - margin)
                y = random.uniform(margin, h - margin)

            x = max(margin, min(w - margin, x))
            y = max(margin, min(h - margin, y))
            valid = True
            for px, py in placed:
                dist = math.sqrt((x - px)**2 + (y - py)**2)
                if dist < min_sep:
                    valid = False
                    break
            if valid:
                return x, y
        return random.uniform(margin, w - margin), random.uniform(margin, h - margin)

    def _spawn(self):
        n = random.randint(config.BALL_COUNT_MIN, config.BALL_COUNT_MAX)
        w, h = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
        r = config.BALL_RADIUS  # Uniform size
        
        placed = []  # Track placed ball positions
        
        if self.spawn_mode == "break_setup" and n >= 8:
            # Seed a triangular-ish cluster near center-right to encourage collisions.
            cx = w * random.uniform(0.58, 0.72)
            cy = h * random.uniform(0.40, 0.60)
            if self.event_profile == "collision_heavy":
                spacing = r * random.uniform(1.95, 2.08)
                rows = min(6, max(4, int(math.sqrt(n)) + 2))
            else:
                spacing = r * 2.15
                rows = min(5, max(3, int(math.sqrt(n)) + 1))
            for row in range(rows):
                for col in range(row + 1):
                    if len(placed) >= n - 1:
                        break
                    x = cx + row * spacing + random.uniform(-0.6, 0.6)
                    y = cy + (col - row / 2.0) * spacing + random.uniform(-0.6, 0.6)
                    x = max(r + 8, min(w - r - 8, x))
                    y = max(r + 8, min(h - r - 8, y))
                    placed.append((x, y))
                if len(placed) >= n - 1:
                    break
            # Leave one ball more isolated for impact shots.
            placed.insert(0, (w * random.uniform(0.18, 0.32), h * random.uniform(0.25, 0.75)))
        elif self.event_profile == "collision_heavy" and self.spawn_mode in ("clustered", "two_cluster") and n >= 8:
            # Keep one striker ball separated so the bot can punch into the cluster.
            placed.insert(0, (w * random.uniform(0.16, 0.30), h * random.uniform(0.22, 0.78)))

        for i in range(n):
            m = config.BALL_MASS

            # Try to find non-overlapping position
            margin = r + 10
            if i < len(placed):
                x, y = placed[i]
            else:
                x, y = self._sample_spawn_position(margin, w, h, r, placed)
                placed.append((x, y))

            body = pymunk.Body(m, pymunk.moment_for_circle(m, 0, r))
            body.position = (x, y)
            body.velocity = (0, 0)  # Start stationary

            shape = pymunk.Circle(body, r)
            shape.elasticity = self.restitution
            shape.friction = self.ball_friction

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
        event_profile = getattr(self.world, "event_profile", "mixed_random")
        if event_profile == "collision_heavy":
            styles = ["breaker", "collider", "chaos", "banker", "scorer", "sniper", "tapper"]
            style_weights = [0.22, 0.22, 0.20, 0.14, 0.08, 0.06, 0.08]
        else:
            styles = ["scorer", "collider", "banker", "chaos", "sniper", "tapper", "breaker"]
            style_weights = [0.22, 0.19, 0.13, 0.18, 0.09, 0.09, 0.10]

        self.style = random.choices(styles, weights=style_weights, k=1)[0]
        profiles = {
            "scorer":  {"wr": (0.05, 0.14), "wb": (0.08, 0.20), "wp": (0.35, 0.55), "wk": (0.06, 0.16), "wt": (0.03, 0.08), "spd": (1.0, 1.45), "acc": (0.01, 0.04), "pow": (0.9, 1.15), "watch": (0.85, 1.05)},
            "collider":{"wr": (0.08, 0.20), "wb": (0.30, 0.48), "wp": (0.10, 0.25), "wk": (0.08, 0.18), "wt": (0.02, 0.06), "spd": (1.0, 1.55), "acc": (0.02, 0.06), "pow": (1.0, 1.35), "watch": (0.70, 0.95)},
            "banker":  {"wr": (0.04, 0.12), "wb": (0.12, 0.24), "wp": (0.12, 0.30), "wk": (0.30, 0.50), "wt": (0.02, 0.06), "spd": (0.9, 1.30), "acc": (0.02, 0.06), "pow": (0.95, 1.25), "watch": (0.85, 1.10)},
            "chaos":   {"wr": (0.18, 0.36), "wb": (0.18, 0.34), "wp": (0.08, 0.22), "wk": (0.08, 0.18), "wt": (0.05, 0.14), "spd": (1.1, 1.70), "acc": (0.04, 0.11), "pow": (1.1, 1.55), "watch": (0.55, 0.85)},
            "sniper":  {"wr": (0.02, 0.08), "wb": (0.08, 0.16), "wp": (0.45, 0.65), "wk": (0.04, 0.10), "wt": (0.02, 0.06), "spd": (0.95, 1.25), "acc": (0.008, 0.03), "pow": (0.85, 1.10), "watch": (0.90, 1.15)},
            "tapper":  {"wr": (0.08, 0.18), "wb": (0.14, 0.26), "wp": (0.14, 0.28), "wk": (0.04, 0.10), "wt": (0.22, 0.40), "spd": (1.0, 1.5), "acc": (0.02, 0.07), "pow": (0.55, 0.90), "watch": (0.70, 0.95)},
            "breaker": {"wr": (0.06, 0.16), "wb": (0.28, 0.42), "wp": (0.06, 0.16), "wk": (0.14, 0.28), "wt": (0.01, 0.05), "spd": (1.0, 1.45), "acc": (0.015, 0.05), "pow": (1.25, 1.75), "watch": (0.95, 1.30)},
        }[self.style]

        # Strategy weights (will be normalized)
        self.w_random = random.uniform(*profiles["wr"])
        self.w_ball = random.uniform(*profiles["wb"])
        self.w_pocket = random.uniform(*profiles["wp"])
        self.w_bank = random.uniform(*profiles["wk"])
        self.w_tap = random.uniform(*profiles["wt"])
        
        # Normalize to sum to 1
        total = self.w_random + self.w_ball + self.w_pocket + self.w_bank + self.w_tap
        self.w_random /= total
        self.w_ball /= total
        self.w_pocket /= total
        self.w_bank /= total
        self.w_tap /= total
        
        # Speed multiplier (0.8 = slower, 1.5 = faster)
        self.speed_mult = random.uniform(*profiles["spd"])

        # Accuracy (lower = more accurate)
        self.accuracy = random.uniform(*profiles["acc"])
        self.power_mult = random.uniform(*profiles["pow"])
        self.watch_mult = random.uniform(*profiles["watch"])
        self.contact_bias = random.uniform(0.0, 1.0)

    def _cluster_center(self, exclude_idx=None):
        balls = []
        for i, (body, _, _) in enumerate(self.world.balls):
            if i == exclude_idx:
                continue
            balls.append((i, body.position.x, body.position.y))
        if not balls:
            return None

        best_idx = None
        best_score = -1.0
        best_center = None
        radius2 = 18.0 * 18.0
        for i, x, y in balls:
            score = 0.0
            count = 0
            sx, sy = 0.0, 0.0
            for j, x2, y2 in balls:
                if i == j:
                    continue
                dx = x2 - x
                dy = y2 - y
                d2 = dx * dx + dy * dy
                if d2 <= radius2:
                    score += 1.0 + (1.0 - min(1.0, d2 / radius2))
                    count += 1
                    sx += x2
                    sy += y2
            if score > best_score:
                best_score = score
                if count > 0:
                    best_center = ((x + sx) / (count + 1), (y + sy) / (count + 1))
                else:
                    best_center = (x, y)
                best_idx = i
        return (best_idx, best_center, best_score)
    
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

        if self.style == "breaker":
            cluster_info = self._cluster_center(exclude_idx=self.ball_idx)
            if cluster_info is not None:
                _, center, score = cluster_info
                dx = center[0] - bx
                dy = center[1] - by
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0.1:
                    aim = humanize(dx, dy, self.accuracy * 0.9)
                    power = min(230, max(90, dist * random.uniform(1.55, 2.05))) * self.power_mult
                    # Punch through clusters more often to create multi-contact sequences.
                    power *= (1.0 + min(0.25, 0.03 * max(0.0, score)))
                    return (aim, power)
        
        # Cumulative thresholds based on personality
        t1 = self.w_random
        t2 = t1 + self.w_ball
        t3 = t2 + self.w_pocket
        t4 = t3 + self.w_bank
        # t5 = 1.0 (tap)
            
        if roll < t1:
            # Random direction
            angle = random.uniform(0, 2 * math.pi)
            power = random.uniform(50, 170) * self.power_mult
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
                base_power = min(175, max(45, dist * 1.5))
                power = base_power * random.uniform(0.75, 1.3) * self.power_mult
                if self.style in ("collider", "chaos") and random.random() < 0.55:
                    power *= random.uniform(1.08, 1.28)
                
                if dist > 0.1:
                    return (humanize(dx, dy), power)
            
            # Fallback to random
            angle = random.uniform(0, 2 * math.pi)
            return ((math.cos(angle), math.sin(angle)), random.uniform(50, 170) * self.power_mult)
            
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
            base_power = min(165, max(50, dist * 1.45))
            power = base_power * random.uniform(0.90, 1.10) * self.power_mult
            
            if dist > 0.1:
                # Almost perfect aim for pocket shots (sniper mode)
                return (humanize(dx, dy, 0.01), power)
                
            angle = random.uniform(0, 2 * math.pi)
            return ((math.cos(angle), math.sin(angle)), random.uniform(50, 170) * self.power_mult)
        
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
                power = min(220, max(70, dist * 1.60)) * random.uniform(0.85, 1.30) * self.power_mult
                
                if dist > 0.1:
                    # More error on bank shots (harder to aim)
                    return (humanize(dx, dy, self.accuracy * 1.5), power)
            
            # Fallback
            angle = random.uniform(0, 2 * math.pi)
            return ((math.cos(angle), math.sin(angle)), random.uniform(55, 180) * self.power_mult)
        
        else:
            # Gentle tap - 10%
            angle = random.uniform(0, 2 * math.pi)
            power = random.uniform(15, 60) * min(self.power_mult, 1.15)  # Very weak / positional
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
                if self.style == "breaker" and len(self.world.balls) >= 3:
                    cluster_info = self._cluster_center()
                    if cluster_info is not None:
                        _, center, _ = cluster_info
                        best_i = 0
                        best_d = -1.0
                        for i, (body, _, _) in enumerate(self.world.balls):
                            dx = body.position.x - center[0]
                            dy = body.position.y - center[1]
                            d2 = dx * dx + dy * dy
                            if d2 > best_d:
                                best_d = d2
                                best_i = i
                        self.ball_idx = best_i
                    else:
                        self.ball_idx = random.randint(0, len(self.world.balls) - 1)
                else:
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
                base_watch = ((0.16 + (self.shot_power / 200.0) * 0.34) / self.speed_mult) * self.watch_mult
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
