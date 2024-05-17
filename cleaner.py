from os import path
from pathlib import Path

import laspy
import numpy as np

from scipy.spatial import ConvexHull


def main(point_cloud_path: str, camera_cloud_path: str, output_file: str, quantile: int = 0,
         buffer_percent: float = 0.1):
    point_cloud_path = str(Path(point_cloud_path).resolve())
    camera_cloud_path = str(Path(camera_cloud_path).resolve())

    cloud_las = laspy.read(point_cloud_path)
    cloud_xyz = cloud_las.xyz

    camera_las = laspy.read(camera_cloud_path)
    camera_xyz = camera_las.xyz

    min_quantile = quantile
    max_quantile = 100 - quantile

    # Get the quantiles of the camera cloud in the x, y, and z directions
    camera_percentiles = np.percentile(camera_xyz, [min_quantile, max_quantile], axis=0)

    # Remove any points in the camera cloud that are outside the 5% and 95% quantiles
    mask = np.all((camera_xyz > camera_percentiles[0]) & (camera_xyz < camera_percentiles[1]), axis=1)
    camera_cloud = camera_xyz[mask]

    # Create a convex hull around the camera cloud
    hull = ConvexHull(camera_cloud)

    # Calculate the 10% buffer distance
    buffer_distance = buffer_percent * np.max(hull.equations[:, -1])

    # Function to check if points are outside the buffered convex hull
    distances = np.dot(cloud_xyz, hull.equations[:, :-1].T) + hull.equations[:, -1]
    mask = np.all(distances <= buffer_distance, axis=1)

    # Filter points from the point cloud
    trimmed_cloud = cloud_xyz[mask]

    # Save the trimmed point cloud to a new LAS file
    trimmed_las = laspy.create(point_format=cloud_las.point_format, file_version=cloud_las.header.version)

    trimmed_las.header.point_format = cloud_las.header.point_format
    trimmed_las.header.scale = cloud_las.header.scale
    trimmed_las.header.offset = cloud_las.header.offset
    trimmed_las.header.x_scale = cloud_las.header.x_scale
    trimmed_las.header.y_scale = cloud_las.header.y_scale
    trimmed_las.header.z_scale = cloud_las.header.z_scale
    trimmed_las.header.x_offset = cloud_las.header.x_offset
    trimmed_las.header.y_offset = cloud_las.header.y_offset
    trimmed_las.header.z_offset = cloud_las.header.z_offset

    trimmed_las.x = trimmed_cloud[:, 0]
    trimmed_las.y = trimmed_cloud[:, 1]
    trimmed_las.z = trimmed_cloud[:, 2]

    trimmed_las.write(output_file)


def point_trimmer(point_cloud: np.ndarray, camera_cloud: np.ndarray) -> np.ndarray:
    """
    This function takes in a point cloud and a camera cloud (points where the camera is located).
    It buffers the camera cloud by 5%, creates a convex hull around the camera cloud, and removes any point outside the
    convex hull from the point cloud. It then returns the cleaned point cloud.
    """


if __name__ == '__main__':
    main("cloud.las", "cam.las", "output.las")
