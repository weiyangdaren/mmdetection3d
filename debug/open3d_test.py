import open3d as o3d
from open3d import geometry
from open3d.visualization import Visualizer


class o3d_visualizer:
    def __init__(self):
        self.o3d_vis = o3d.visualization.VisualizerWithKeyCallback()
        self.o3d_vis.register_key_callback(glfw_key_escape, self.escape_callback)
        self.o3d_vis.register_key_action_callback(glfw_key_space,
                                                self.space_action_callback)
        self.o3d_vis.register_key_callback(glfw_key_right, self.right_callback)

        self.o3d_vis.create_window()
        self.view_control = self.o3d_vis.get_view_control()

    def escape_callback(self, vis):
        self.o3d_vis.clear_geometries()
        self.o3d_vis.destroy_window()
        self.o3d_vis.close()
        self._clear_o3d_vis()
        sys.exit(0)
    
    def space_action_callback(self, vis, action, mods):
        if action == 1:
            if self.flag_pause:
                print_log(
                    'Playback continued, press [SPACE] to pause.',
                    logger='current')
            else:
                print_log(
                    'Playback paused, press [SPACE] to continue.',
                    logger='current')
            self.flag_pause = not self.flag_pause
        return True

    def right_callback(self, vis):
        self.flag_next = True
        return False

if __name__ == '__main__':
    o3d_vis = o3d_visualizer()
    print('done')