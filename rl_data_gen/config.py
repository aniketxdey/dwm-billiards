"""
Neural Newton - Central Configuration
All magic numbers in one place.
"""

# === Display ===
SCREEN_WIDTH = 128  # 16:9 would be 128x72, using 128x72
SCREEN_HEIGHT = 72
FPS = 30

# === Physics (Pymunk) ===
# Billiard/Pool style - top-down view, no gravity
GRAVITY = 0  # Top-down billiard table
FRICTION = 0.95  # Higher friction - balls slow down faster
RESTITUTION = 0.9  # Bouncy cushions
DAMPING = 0.98  # Balls slow down over time

# Legacy random ranges (for Phase 2)
GRAVITY_MIN = -1000
GRAVITY_MAX = 1000
FRICTION_MIN = 0.3
FRICTION_MAX = 0.9
RESTITUTION_MIN = 0.5
RESTITUTION_MAX = 0.95

# === Balls ===
BALL_RADIUS = 3  # Smaller uniform size
BALL_RADIUS_MIN = 3  # Legacy
BALL_RADIUS_MAX = 3  # Legacy
BALL_MASS = 1.0

# Ball count - v3 biases toward denser scenes to increase collision frequency.
BALL_COUNT = 8
BALL_COUNT_MIN = 10
BALL_COUNT_MAX = 16

def calculate_ball_count(avg_radius=None):
    """Simple random ball count."""
    import random
    return random.randint(BALL_COUNT_MIN, BALL_COUNT_MAX)

# === Ghost Hand (Mouse) ===
MOUSE_RADIUS = 5
MOUSE_GRAB_SPRING_STIFFNESS = 500
MOUSE_GRAB_SPRING_DAMPING = 50
MOUSE_COLLISION_GROUP = 1  # For filtering

# === Data Generation ===
EPISODE_FRAMES_MIN = 450  # 15 seconds at 30fps
EPISODE_FRAMES_MAX = 600  # 20 seconds at 30fps

# === Ghost Hand Modes ===
MODE_DURATION_MIN = 0.5  # seconds
MODE_DURATION_MAX = 2.0  # seconds

# === Normalization (for model input) ===
def normalize_mouse(x, y):
    """Normalize mouse coords from [0, SCREEN] to [-1, 1]"""
    nx = (x / SCREEN_WIDTH) * 2 - 1
    ny = (y / SCREEN_HEIGHT) * 2 - 1
    return nx, ny

def denormalize_mouse(nx, ny):
    """Denormalize mouse coords from [-1, 1] to [0, SCREEN]"""
    x = (nx + 1) / 2 * SCREEN_WIDTH
    y = (ny + 1) / 2 * SCREEN_HEIGHT
    return x, y

def normalize_gravity(g):
    """Normalize gravity from [GRAVITY_MIN, GRAVITY_MAX] to [-1, 1]"""
    return (g - GRAVITY_MIN) / (GRAVITY_MAX - GRAVITY_MIN) * 2 - 1

def denormalize_gravity(ng):
    """Denormalize gravity from [-1, 1] to [GRAVITY_MIN, GRAVITY_MAX]"""
    return (ng + 1) / 2 * (GRAVITY_MAX - GRAVITY_MIN) + GRAVITY_MIN

# === VAE ===
LATENT_SCALE = 0.18215  # Standard SD VAE scaling

# === Model ===
CONTEXT_LENGTH = 4  # Number of history frames

# === Paths ===
DATASET_RAW_DIR = "dataset_raw"
DATASET_LATENTS_DIR = "dataset_latents"
CHECKPOINTS_DIR = "checkpoints"
