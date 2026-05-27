from __future__ import annotations

import argparse
import time

from franka_mujoco.env import FrankaPandaEnv


def run_viewer(duration: float | None = None, headless: bool = False, realtime: bool = True) -> None:
    env = FrankaPandaEnv()

    if headless:
        _run_loop(env, duration=duration, realtime=False, viewer=None)
        return

    from mujoco import viewer as mujoco_viewer  # type: ignore

    with mujoco_viewer.launch_passive(env.model, env.data) as viewer:
        _run_loop(env, duration=duration, realtime=realtime, viewer=viewer)


def _run_loop(env: FrankaPandaEnv, duration: float | None, realtime: bool, viewer) -> None:
    start_time = env.time
    next_wall_time = time.time()

    while True:
        if duration is not None and env.time - start_time >= duration:
            break
        if viewer is not None and not viewer.is_running():
            break

        env.step()

        if viewer is not None:
            viewer.sync()

        if realtime:
            next_wall_time += env.timestep
            sleep_s = next_wall_time - time.time()
            if sleep_s > 0.0:
                time.sleep(sleep_s)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Open the Franka Panda MuJoCo viewer.")
    parser.add_argument("--headless", action="store_true", help="Load and step without opening a viewer.")
    parser.add_argument("--duration", type=float, default=None, help="Optional run duration in seconds.")
    parser.add_argument("--no-realtime", action="store_true", help="Run as fast as MuJoCo can step.")
    args = parser.parse_args(argv)

    run_viewer(
        duration=args.duration,
        headless=args.headless,
        realtime=not args.no_realtime,
    )


if __name__ == "__main__":
    main()
