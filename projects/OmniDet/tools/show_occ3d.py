import numpy as np
import mmengine
from pathlib import Path
import mayavi.mlab as mlab
from traits.api import HasTraits, Button
from traitsui.api import View, Item
from pyface.api import GUI


COLOR_MAP = np.array([
    [150, 240,  80, 255],  # terrain     light green
    [180, 180, 180, 255],  # ground      gray
    [255,   0, 255, 255],  # roads       dark pink
    [ 75,   0,  75, 255],  # sidewalks   dark purple
    [230, 230, 250, 255],  # buildings   white
    [255, 120,  50, 255],  # guardrail   orangey
    [135,  60,   0, 255],  # poles       brown
    [  0, 175,   0, 255],  # vegetation  green
    [255,   0,   0, 255],  # pedestrian  red
    [255, 192, 203, 255],  # cyclist     pink
    [  0, 150, 245, 255],  # car         blue
    [160,  32, 240, 255],  # truck       purple
    [255, 255,   0, 255],  # bus         yellow
    [200, 180,   0, 255],  # misc        dark orange
    [150, 220, 248, 255],  # background  light blue
    [255, 255, 255, 255],  # free        white
], dtype=np.uint8)

# mlab.options.offscreen = True


class ViewSaver(HasTraits):
    save_button = Button("Save View Params")

    def _save_button_fired(self):
        
        view_params = {
            'azimuth': mlab.view()[0],
            'elevation': mlab.view()[1],
            'distance': mlab.view()[2],
            'focalpoint': mlab.view()[3]
        }
        print(view_params)
        mmengine.dump(view_params, 'view_params.pkl')

        print("Save view params done!")
        
    traits_view = View(
        Item('save_button'),
        title="ViewSaver",
        width=300,
        height=100,
        resizable=True
    )


class OccVisualizer:
    def __init__(self,
                 voxel_size,
                 pc_range,
                 fig_size=(1920, 1080),
                 file_dir=None,
                 save_dir=None,
                 view_params_path=None):
        self.voxel_size = np.array(voxel_size)
        self.pc_range = np.array(pc_range)
        self.fig_size = fig_size
        self.file_dir = Path(file_dir)
        self.save_dir = Path(save_dir)
        Path(self.save_dir).mkdir(exist_ok=True)
        self.view_params = self.load_view_params(view_params_path)

    def init_fig(self):
        return mlab.figure(size=self.fig_size, bgcolor=(1, 1, 1))
    
    def get_file_list(self):
        return [x for x in Path(self.file_dir).iterdir() if x.suffix == '.npz']

    def load_voxel(self, file_path):
        return np.load(file_path)['voxel'].astype(np.float32)
    
    def load_view_params(self, file_path):
        if file_path is not None:
            return mmengine.load(file_path)
        else:
            return None

    def process_voxel(self, voxels):
        voxels[:, :3] = (voxels[:, :3] + 0.5) * self.voxel_size
        voxels[:, 0] += self.pc_range[0]
        voxels[:, 1] += self.pc_range[1]
        voxels[:, 2] += self.pc_range[2]
        return voxels

    def draw_scene(self, file_path, figure=None):
        voxels = self.load_voxel(file_path)
        voxels = self.process_voxel(voxels)
        
        plt_plot_fov = mlab.points3d(
            voxels[:, 0],
            voxels[:, 1],
            voxels[:, 2],
            voxels[:, 3],
            figure=figure,
            colormap="viridis",
            scale_factor=self.voxel_size[0] - 0.05*self.voxel_size[0],
            mode="cube",
            opacity=1.0,
            vmin=0,
            vmax=len(COLOR_MAP) - 1,
        )

        plt_plot_fov.glyph.scale_mode = "scale_by_vector"
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = COLOR_MAP

        if self.view_params is not None:
            mlab.view(**self.view_params)
    
    def save_view_param(self):
        file_path = r'occ\00027000.npz'
        self.draw_scene(file_path)
        saver = ViewSaver()
        saver.edit_traits(kind='live') 
        mlab.show()

        mlab.close(all=True)
        GUI().process_events()
    
    def save_image(self):
        file_list = self.get_file_list()
        figure = self.init_fig()

        for file_path in file_list:
            self.draw_scene(file_path, figure)
            save_path = self.save_dir / (file_path.stem + '.png')
            mlab.savefig(str(save_path))
            mlab.clf()
            print(f'Finish save image in {save_path}')


if __name__ == '__main__':
    occ_vis = OccVisualizer(
        voxel_size=[0.4,0.4,0.4],
        pc_range=[-40,-40,-2.4,40,40,4],
        fig_size=[1920, 1080],
        file_dir=r"\\172.22.143.207\Dataset\CarlaCollection\train-Town01_Opt-ClearNoon-2024_09_23_11_44_45\ego0\label_occ",
        save_dir='occ_images',
        view_params_path='view_params.pkl'
        )
    occ_vis.save_image()
    
        

