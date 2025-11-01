from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


def main():
    print("Hello from reachy-robot!")

    with ReachyMini() as r:
        r.goto_target(head=create_head_pose(z=10, roll=15, degrees=True, mm=True), duration=2.0)
        r.goto_target(head=create_head_pose(), duration=2.0)


if __name__ == "__main__":
    main()
