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
    cloud_points = cloud_las.points.copy()
    cloud_xyz = np.vstack((cloud_las.x, cloud_las.y, cloud_las.z)).T

    camera_las = laspy.read(camera_cloud_path)
    camera_xyz = np.vstack((camera_las.x, camera_las.y, camera_las.z)).T

    min_quantile = quantile
    max_quantile = 100 - quantile

    # Get the quantiles of the camera cloud in the x, y, and z directions
    camera_percentiles = np.percentile(camera_xyz, [min_quantile, max_quantile], axis=0)

    # Remove any points in the camera cloud that are outside the specified quantiles
    mask = np.all((camera_xyz >= camera_percentiles[0]) & (camera_xyz <= camera_percentiles[1]), axis=1)
    camera_cloud = camera_xyz[mask]

    # Create a convex hull around the filtered camera cloud
    hull = ConvexHull(camera_cloud)

    # Calculate the buffer distance
    buffer_distance = buffer_percent * np.max(hull.equations[:, -1])

    # Function to check if points are within the buffered convex hull
    def is_within_hull(point, hull, buffer_distance):
        return np.all(np.dot(hull.equations[:, :-1], point.T) + hull.equations[:, -1] <= buffer_distance)

    # Filter points from the point cloud
    mask = np.array([is_within_hull(point, hull, buffer_distance) for point in cloud_xyz])
    trimmed_points = cloud_points[mask]

    # Create a new LAS file for the trimmed point cloud
    trimmed_las = laspy.create(point_format=cloud_las.header.point_format, file_version=cloud_las.header.version)

    # Copy header information
    trimmed_las.header = cloud_las.header
    trimmed_las.points = trimmed_points

    trimmed_las.write(output_file)



if __name__ == '__main__':
    main("cloud.las", "cam.las", "output.las")
