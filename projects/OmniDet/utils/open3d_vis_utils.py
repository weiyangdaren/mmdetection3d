"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def draw_scenes_v2(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, point_labels=None, draw_origin=True):
    # draw origin
    draw_list = []
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        draw_list.append(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.zeros((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if point_labels is not None:
        num_classes = len(np.unique(point_labels))
        colors = np.zeros((len(points), 3))
        for i in range(num_classes):
            class_indices = np.where(point_labels == i)[0]
            colors[class_indices] = np.random.rand(3)
        pts.colors = open3d.utility.Vector3dVector(colors)
    draw_list.append(pts)

    if gt_boxes is not None:
        gt_lines = draw_box_v2(gt_boxes, (1, 0, 0))
        draw_list.extend(gt_lines)
    if ref_boxes is not None:
        pred_lines = draw_box_v2(ref_boxes, (0, 1, 0), ref_labels, score=ref_scores)
        draw_list.extend(pred_lines)

    # 设置相机的视角
    # lookat = [25, 3, 0]  # 相机的焦点位置
    # eye = [25, 3, 25]  # 相机的位置
    # up = [0, 1, 0]  # 相机的上方向
    bg_color = [0.96078, 0.97255, 0.96863, 1.0]
    open3d.visualization.draw(draw_list, raw_mode=True, show_skybox=False, line_width=5, point_size=6, bg_color=bg_color)
    # open3d.visualization.draw(draw_list, lookat=lookat, eye=eye, up=up, raw_mode=True, show_skybox=False, line_width=5, point_size=3, bg_color=bg_color)



def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis


def draw_box_v2(gt_boxes, color=(0, 1, 0), ref_labels=None, name=None, score=None):
    lines = []
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])
        lines.append(line_set)

        if score is not None:
            corners = box3d.get_box_points()
            text = "{:.2f}".format(score[i])  # Format score to 2 decimal places
            # Create text mesh
            text_mesh = open3d.t.geometry.TriangleMesh.create_text(
                text,
                depth=0.01,  # Set smaller depth for smaller text
                float_dtype=open3d.core.Dtype.Float32,
                int_dtype=open3d.core.Dtype.Int32,
                device=open3d.core.Device("CPU:0")
            )
            # Place the text at the position of corner[5]
            text_mesh.translate(corners[5] + np.array([0, 0.1, 0]))  # Adjust the height if needed
            lines.append(text_mesh)
         
    return lines
