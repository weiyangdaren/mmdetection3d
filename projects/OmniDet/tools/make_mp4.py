import cv2
from tqdm import tqdm
from pathlib import Path

def create_video_from_png(input_dir, output_path, fps=30, resolution=(1920, 1080)):
    input_dir = Path(input_dir)
    images = [x.name for x in input_dir.iterdir() if x.suffix=='.png']
    images.sort()  

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    for image_name in tqdm(images):
        img_path = input_dir / image_name
        frame = cv2.imread(img_path)
        
        if frame is not None:
            # 调整图像尺寸到目标分辨率
            resized_frame = cv2.resize(frame, resolution)
            video_writer.write(resized_frame)
        else:
            print(f"警告：无法读取图像 {image_name}")

    video_writer.release()
    print(f"视频已保存至 {output_path}")

# 使用示例
if __name__ == "__main__":
    create_video_from_png(
        input_dir="occ_images",
        output_path="output_video.mp4",
        fps=10,
        resolution=(1280, 720)
    )
