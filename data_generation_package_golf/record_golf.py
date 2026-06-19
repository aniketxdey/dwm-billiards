#!/usr/bin/env python3
"""
Neural Golf - Top-down 2D mini-golf recording.

A single ball on a felt course with a cup (hole), border cushions, and a few
static bumpers. An autoplay bot lines up the ball toward the cup and putts.
When the ball is captured by the cup, it re-tees at a new spot with a new cup.

The world/bot intentionally mirror the billiards package interface so the
shared shard generator and downstream VAE/world-model pipeline work unchanged.
"""

import math
import random

import cv2
import numpy as np
import pymunk

import config


# Colors (RGB)
GRASS_LIGHT = (62, 142, 70)    # mowed stripe A
GRASS_DARK = (44, 120, 54)     # mowed stripe B
ROUGH_COLOR = (33, 88, 40)     # darker border rough
RAIL_COLOR = (120, 84, 48)     # wooden rail
HOLE_COLOR = (16, 16, 16)      # cup
FLAG_POLE = (230, 230, 230)
FLAG_CLOTH = (220, 45, 45)
BALL_COLOR = (245, 245, 245)   # white golf ball
BALL_SHADE = (180, 180, 180)
BUMPER_COLOR = (140, 140, 150)  # grey rock
BUMPER_EDGE = (70, 70, 80)


def _rand_point(margin):
    w, h = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
    return (random.uniform(margin, w - margin), random.uniform(margin, h - margin))


class GolfWorld:
    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = config.DAMPING

        self.ball = None
        self.ball_radius = config.BALL_RADIUS
        self.bumpers = []  # list of (x, y, r)
        self.hole = (0.0, 0.0)
        self.sunk = 0  # cups completed

        self._walls()
        self._spawn_bumpers()
        self.hole = self._new_hole()
        self._spawn_ball()

    # --- construction -----------------------------------------------------
    def _walls(self):
        t = 3
        w, h = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
        walls = [
            ((t, t), (w - t, t)),
            ((t, h - t), (w - t, h - t)),
            ((t, t), (t, h - t)),
            ((w - t, t), (w - t, h - t)),
        ]
        for a, b in walls:
            seg = pymunk.Segment(self.space.static_body, a, b, t)
            seg.elasticity = config.RESTITUTION
            seg.friction = config.FRICTION
            self.space.add(seg)

    def _spawn_bumpers(self):
        n = random.randint(config.BUMPER_COUNT_MIN, config.BUMPER_COUNT_MAX)
        r = config.BUMPER_RADIUS
        for _ in range(n):
            for _attempt in range(50):
                x, y = _rand_point(margin=r + 14)
                if all(math.hypot(x - bx, y - by) > (r + br + 8) for bx, by, br in self.bumpers):
                    break
            shape = pymunk.Circle(self.space.static_body, r, offset=(x, y))
            shape.elasticity = 0.95
            shape.friction = 0.4
            self.space.add(shape)
            self.bumpers.append((x, y, r))

    def _clear_of_bumpers(self, x, y, pad):
        return all(math.hypot(x - bx, y - by) > (br + pad) for bx, by, br in self.bumpers)

    def _new_hole(self):
        for _attempt in range(100):
            x, y = _rand_point(margin=16)
            if self._clear_of_bumpers(x, y, pad=config.BUMPER_RADIUS + 8):
                return (x, y)
        return _rand_point(margin=16)

    def _spawn_ball(self):
        r = self.ball_radius
        hx, hy = self.hole
        x, y = _rand_point(margin=r + 12)
        for _attempt in range(100):
            x, y = _rand_point(margin=r + 12)
            far_from_hole = math.hypot(x - hx, y - hy) > 35
            if far_from_hole and self._clear_of_bumpers(x, y, pad=r + 6):
                break

        body = pymunk.Body(config.BALL_MASS, pymunk.moment_for_circle(config.BALL_MASS, 0, r))
        body.position = (x, y)
        body.velocity = (0, 0)
        shape = pymunk.Circle(body, r)
        shape.elasticity = config.RESTITUTION
        shape.friction = config.FRICTION
        self.space.add(body, shape)
        self.ball = body

    def _remove_ball(self):
        if self.ball is not None:
            for shape in list(self.ball.shapes):
                self.space.remove(shape)
            self.space.remove(self.ball)
            self.ball = None

    # --- control ----------------------------------------------------------
    def putt(self, velocity):
        if self.ball is not None:
            self.ball.velocity = velocity

    def ball_speed(self):
        if self.ball is None:
            return 0.0
        v = self.ball.velocity
        return math.hypot(v.x, v.y)

    # --- simulation -------------------------------------------------------
    def step(self, dt):
        self.space.step(dt)

        if self.ball is None:
            self.hole = self._new_hole()
            self._spawn_ball()
            return

        bx, by = self.ball.position.x, self.ball.position.y
        hx, hy = self.hole
        if math.hypot(bx - hx, by - hy) < config.HOLE_RADIUS and self.ball_speed() < config.SINK_SPEED:
            self.sunk += 1
            self._remove_ball()
            self.hole = self._new_hole()
            self._spawn_ball()

    # --- rendering --------------------------------------------------------
    def render(self, scale):
        w, h = config.SCREEN_WIDTH * scale, config.SCREEN_HEIGHT * scale
        img = np.empty((h, w, 3), dtype=np.uint8)

        # mowed stripes
        stripe = max(6 * scale, 1)
        for y0 in range(0, h, stripe):
            color = GRASS_LIGHT if (y0 // stripe) % 2 == 0 else GRASS_DARK
            img[y0:y0 + stripe] = color[::-1]

        # rough border + wooden rail
        rough = 4 * scale
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), ROUGH_COLOR[::-1], rough)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), RAIL_COLOR[::-1], 2 * scale)

        # bumpers
        for bx, by, br in self.bumpers:
            c = (int(bx * scale), int(by * scale))
            cv2.circle(img, c, int(br * scale) + 1, BUMPER_EDGE[::-1], -1, cv2.LINE_AA)
            cv2.circle(img, c, int(br * scale), BUMPER_COLOR[::-1], -1, cv2.LINE_AA)

        # hole + flag
        hx, hy = self.hole
        hc = (int(hx * scale), int(hy * scale))
        cv2.circle(img, hc, int(config.HOLE_RADIUS * scale), HOLE_COLOR[::-1], -1, cv2.LINE_AA)
        pole_top = (hc[0], hc[1] - 12 * scale)
        cv2.line(img, hc, pole_top, FLAG_POLE[::-1], max(1, scale), cv2.LINE_AA)
        flag_pts = np.array([
            pole_top,
            (pole_top[0] + 7 * scale, pole_top[1] + 3 * scale),
            (pole_top[0], pole_top[1] + 6 * scale),
        ], dtype=np.int32)
        cv2.fillPoly(img, [flag_pts], FLAG_CLOTH[::-1], cv2.LINE_AA)

        # ball
        if self.ball is not None:
            x = int(self.ball.position.x * scale)
            y = int(self.ball.position.y * scale)
            r = int(self.ball_radius * scale)
            cv2.circle(img, (x, y), r + 1, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), r, BALL_COLOR[::-1], -1, cv2.LINE_AA)
            cv2.circle(img, (x - max(1, r // 3), y - max(1, r // 3)),
                       max(1, r // 2), BALL_SHADE[::-1], -1, cv2.LINE_AA)

        # score (cups sunk)
        cv2.putText(img, str(self.sunk), (w - 30, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)

        return img


class GolfBot:
    """
    Autoplay golfer. Waits for the ball to settle, then putts toward the cup
    with a randomized accuracy/power "personality" for varied trajectories.
    """

    def __init__(self, world):
        self.world = world
        self.cooldown = random.uniform(0.2, 0.5)
        self.did_shoot = False
        self.shot_force = (0.0, 0.0)
        self._randomize_personality()

    def _randomize_personality(self):
        self.accuracy = random.uniform(0.03, 0.14)   # aim angle std (radians)
        self.power_mult = random.uniform(0.85, 1.25)  # systematic over/under hit
        self.miss_prob = random.uniform(0.05, 0.20)   # chance of a deliberately off shot

    def _aim(self):
        bx, by = self.world.ball.position.x, self.world.ball.position.y
        hx, hy = self.world.hole
        dx, dy = hx - bx, hy - by
        dist = max(math.hypot(dx, dy), 1e-3)

        angle = math.atan2(dy, dx) + random.gauss(0, self.accuracy)
        power = min(config.POWER_MAX, max(config.POWER_MIN, dist * config.POWER_PER_PX))
        power *= self.power_mult * random.uniform(0.9, 1.1)

        if random.random() < self.miss_prob:
            # occasional wild/under shot for diversity
            angle += random.uniform(-0.6, 0.6)
            power *= random.uniform(0.4, 1.4)

        return (math.cos(angle) * power, math.sin(angle) * power)

    def update(self, dt):
        self.did_shoot = False
        self.shot_force = (0.0, 0.0)
        if self.world.ball is None:
            return

        self.cooldown -= dt
        if self.world.ball_speed() < 3.0 and self.cooldown <= 0:
            fx, fy = self._aim()
            self.world.putt((fx, fy))
            self.did_shoot = True
            self.shot_force = (fx, fy)
            self.cooldown = random.uniform(0.5, 1.3)


def main():
    print("Recording mini-golf demo...")
    world = GolfWorld()
    bot = GolfBot(world)

    scale = 8
    w, h = config.SCREEN_WIDTH * scale, config.SCREEN_HEIGHT * scale
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("golf_demo.mp4", fourcc, config.FPS, (w, h))

    dt = 1.0 / config.FPS
    for i in range(20 * config.FPS):
        bot.update(dt)
        world.step(dt)
        out.write(world.render(scale))
        if i % 100 == 0:
            print(f"  {i} frames | cups sunk: {world.sunk}")
    out.release()
    print("Done -> golf_demo.mp4")


if __name__ == "__main__":
    main()
