"""
Neural Golf - Central Configuration
Top-down 2D mini-golf. Mirrors the billiards data package so the same
VAE / world-model / inference pipeline consumes it unchanged.
"""

# === Display ===
SCREEN_WIDTH = 128  # keep 128x72 so VAE latent stays 9x16 (8x downsample)
SCREEN_HEIGHT = 72
FPS = 30

# === Physics (Pymunk) ===
# Top-down course, no gravity. Strong damping so the ball rolls and stops
# like a putt (pymunk damping is "fraction of velocity retained per second").
GRAVITY = 0
FRICTION = 0.9          # surface friction on contacts
RESTITUTION = 0.7       # wall/bumper bounciness
DAMPING = 0.30          # ~70% of velocity lost per second -> ball settles in ~2s

# Legacy random ranges (unused, kept for schema/config parity)
GRAVITY_MIN = -1000
GRAVITY_MAX = 1000
FRICTION_MIN = 0.3
FRICTION_MAX = 0.9
RESTITUTION_MIN = 0.5
RESTITUTION_MAX = 0.95

# === Ball ===
# radius 4 (not 3): a larger ball is far easier for the 8x-downsampling VAE to
# reconstruct, which is the main accuracy gate for golf-like simulation.
BALL_RADIUS = 4
BALL_RADIUS_MIN = 4
BALL_RADIUS_MAX = 4
BALL_MASS = 1.0

# === Hole / cup ===
HOLE_RADIUS = 5
SINK_SPEED = 55.0       # ball is captured only if slow enough near the cup

# === Course obstacles (static bumpers / "rocks") ===
BUMPER_COUNT_MIN = 1
BUMPER_COUNT_MAX = 3
BUMPER_RADIUS = 4

# === Shot calibration (bot) ===
# distance(px) -> launch speed(px/s); tuned for DAMPING above so the ball
# tends to reach the hole region.
POWER_PER_PX = 1.5
POWER_MIN = 35.0
POWER_MAX = 200.0

# Legacy ball-count names some shared code references; golf has one ball.
BALL_COUNT = 1
BALL_COUNT_MIN = 1
BALL_COUNT_MAX = 1

def calculate_ball_count(avg_radius=None):
    return 1

# === Data Generation ===
EPISODE_FRAMES_MIN = 450
EPISODE_FRAMES_MAX = 600

# === Normalization helpers (kept for parity with billiards config) ===
def normalize_mouse(x, y):
    nx = (x / SCREEN_WIDTH) * 2 - 1
    ny = (y / SCREEN_HEIGHT) * 2 - 1
    return nx, ny

def denormalize_mouse(nx, ny):
    x = (nx + 1) / 2 * SCREEN_WIDTH
    y = (ny + 1) / 2 * SCREEN_HEIGHT
    return x, y

# === VAE / Model parity ===
LATENT_SCALE = 0.18215
CONTEXT_LENGTH = 4

# === Paths ===
DATASET_RAW_DIR = "dataset_raw"
DATASET_LATENTS_DIR = "dataset_latents"
CHECKPOINTS_DIR = "checkpoints"
