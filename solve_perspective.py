import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

def extract_data(filepath):
    with open(filepath, "rb") as f:
        # Array 1: Positions (3 floats per point)
        f.seek(381888)
        pos_data = f.read(628 * 12)

        # Array 2: Colors/Tex (3 floats per point)
        f.seek(389424)
        col_data = f.read(628 * 12)

        # Array 3: Indices? Or just another float array? (1 float per point)
        f.seek(396960)
        idx_data = f.read(628 * 4)

    positions = []
    for i in range(628):
        x, y, z = struct.unpack_from("3f", pos_data, i * 12)
        positions.append((x, y, z))

    colors = []
    for i in range(628):
        r, g, b = struct.unpack_from("3f", col_data, i * 12)
        colors.append((r, g, b))

    extras = []
    for i in range(628):
        val = struct.unpack_from("f", idx_data, i * 4)[0]
        extras.append(val)

    return np.array(positions), np.array(colors), np.array(extras)

def rotate_points(points, angle_degrees, axis='y'):
    theta = np.radians(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)

    if axis == 'y':
        # Rotate around Y axis
        R = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis == 'z':
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    return np.dot(points, R.T)

def main():
    exe_path = "perspective.exe"
    if not os.path.exists(exe_path):
        # Check if it is in /tmp/file_attachments/2/perspective.exe
        alt_path = "/tmp/file_attachments/2/perspective.exe"
        if os.path.exists(alt_path):
            exe_path = alt_path
        else:
            print("perspective.exe not found.")
            return

    positions, colors, extras = extract_data(exe_path)

    # Filter for points that look like the "signal".
    mask = (positions[:, 0] > -8) & (positions[:, 0] < 4) & \
           (positions[:, 1] > -2) & (positions[:, 1] < 2) & \
           (positions[:, 2] > 0) & (positions[:, 2] < 10)

    filtered_pos = positions[mask]
    filtered_col = colors[mask]

    # Rotate around Y to straighten the line
    # Slope = -2/3. Angle = atan(-2/3) ~ -33.69 degrees.
    angle = np.degrees(np.arctan(2/3))

    rotated_pos = rotate_points(filtered_pos, -angle, axis='y')

    fig = plt.figure(figsize=(20, 5))

    # Rotated XY View
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(rotated_pos[:, 0], rotated_pos[:, 1], c=filtered_col, marker='.')
    ax.set_title('Rotated XY View (Flag)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')

    output_file = 'solved_flag.png'
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved solution image to {output_file}")

if __name__ == "__main__":
    main()
