from os import path
from pathlib import Path

import laspy
import numpy as np
import click

from scipy.spatial import ConvexHull


@click.command("Clean", help="Filter a point cloud based on a camera cloud.")
@click.argument('point_cloud_path', type=click.Path(exists=True))
@click.argument('camera_cloud_path', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--quantile', default=0, help='Quantile of camera coordinates to filter cloud points')
@click.option('--buffer_percent', default=0.1, help='Buffer percentage to filter point cloud points')
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

    max_distance = 0
    for i in range(len(hull.vertices)):
        for j in range(i + 1, len(hull.vertices)):
            dist = np.linalg.norm(camera_cloud[hull.vertices[i]] - camera_cloud[hull.vertices[j]])
            if dist > max_distance:
                max_distance = dist

    # Calculate the buffer distance
    buffer_distance = buffer_percent * max_distance

    # Function to check if points are within the buffered convex hull
    def is_within_hull(point, convex_hull, buf_distance):
        return np.all(np.dot(convex_hull.equations[:, :-1], point.T) + convex_hull.equations[:, -1] <= buf_distance)

    # Filter points from the point cloud
    mask = np.array([is_within_hull(point, hull, buffer_distance) for point in cloud_xyz])
    trimmed_points = cloud_points[mask]

    # Create a new LAS file for the trimmed point cloud
    trimmed_las = laspy.create(point_format=cloud_las.header.point_format, file_version=cloud_las.header.version)

    # Copy header information
    trimmed_las.header = cloud_las.header
    trimmed_las.points = trimmed_points

    trimmed_las.write(output_file)
