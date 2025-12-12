import argparse
import cv2
import glob
import mimetypes
import numpy as np
import os
import shutil
import subprocess
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from os import path as osp
from tqdm import tqdm

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

try:
    import ffmpeg
except ImportError:
    import pip
    pip.main(['install', '--user', 'ffmpeg-python'])
    import ffmpeg

# ==============================================================================
#  1. OPTIMIZATION KERNELS (Optical Flow & Warping)
# ==============================================================================

def get_scene_change_score(img1_gray, img2_gray):
    """
    Calculates score to detect scene changes.
    Uses small resize for speed. Returns mean absolute difference.
    """
    h, w = img1_gray.shape
    # Downscale for instant comparison
    if w > 256:
        new_w = 256
        new_h = int(h * 256 / w)
        img1_small = cv2.resize(img1_gray, (new_w, new_h))
        img2_small = cv2.resize(img2_gray, (new_w, new_h))
        diff = cv2.absdiff(img1_small, img2_small)
    else:
        diff = cv2.absdiff(img1_gray, img2_gray)
    
    return np.mean(diff)

def get_optical_flow(prev_gray, curr_gray):
    """
    Calculates dense optical flow.
    Downscales input first to make CPU calculation faster than GPU inference.
    """
    h, w = prev_gray.shape
    
    # Critical: Downscale significantly for speed. 
    # Processing flow at 1080p on CPU is too slow. 320p is good enough for warping.
    flow_render_w = 320 
    flow_render_h = int(h * flow_render_w / w)
    
    prev_small = cv2.resize(prev_gray, (flow_render_w, flow_render_h))
    curr_small = cv2.resize(curr_gray, (flow_render_w, flow_render_h))
    
    # Calculate flow on small image
    flow_small = cv2.calcOpticalFlowFarneback(prev_small, curr_small, None, 
                                        pyr_scale=0.5, levels=3, winsize=15, 
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    
    # Scale flow back up to original resolution
    # 1. Resize the flow map
    flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 2. Scale the flow values (magnitude)
    scale_x = w / flow_render_w
    scale_y = h / flow_render_h
    flow[..., 0] *= scale_x
    flow[..., 1] *= scale_y
    
    return flow

def warp_image(img, flow):
    """Warps an image based on the flow field (Backward Warping)."""
    h, w = flow.shape[:2]
    
    # Generate grid
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Subtract flow for backward warping
    map_x = (grid_x - flow[..., 0]).astype(np.float32)
    map_y = (grid_y - flow[..., 1]).astype(np.float32)
    
    # Remap
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped

# ==============================================================================
#  2. VIDEO IO UTILITIES (Unchanged)
# ==============================================================================

def get_video_meta_info(video_path):
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    if 'nb_frames' in video_streams[0]:
        ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    else:
        # Fallback estimation
        if 'duration' in video_streams[0]:
            duration = float(video_streams[0]['duration'])
        elif 'duration' in probe.get('format', {}):
            duration = float(probe['format']['duration'])
        else:
            duration = 0
        ret['nb_frames'] = int(duration * ret['fps']) if duration > 0 else 999999
    return ret

def get_sub_video(args, num_process, process_idx):
    if num_process == 1:
        return args.input
    meta = get_video_meta_info(args.input)
    duration = int(meta['nb_frames'] / meta['fps'])
    part_time = duration // num_process
    print(f'duration: {duration}, part_time: {part_time}')
    os.makedirs(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'), exist_ok=True)
    out_path = osp.join(args.output, f'{args.video_name}_inp_tmp_videos', f'{process_idx:03d}.mp4')
    cmd = [
        args.ffmpeg_bin, f'-i {args.input}', '-ss', f'{part_time * process_idx}',
        f'-to {part_time * (process_idx + 1)}' if process_idx != num_process - 1 else '', '-async 1', out_path, '-y'
    ]
    print(' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True)
    return out_path

class Reader:
    def __init__(self, args, total_workers=1, worker_idx=0):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.paths = []
        self.audio = None
        self.input_fps = None
        if self.input_type.startswith('video'):
            video_path = get_sub_video(args, total_workers, worker_idx)
            self.stream_reader = (
                ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24',
                                                loglevel='error').run_async(
                                                    pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
            meta = get_video_meta_info(video_path)
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio']
            self.nb_frames = meta['nb_frames']
        else:
            if self.input_type.startswith('image'):
                self.paths = [args.input]
            else:
                paths = sorted(glob.glob(os.path.join(args.input, '*')))
                tot_frames = len(paths)
                num_frame_per_worker = tot_frames // total_workers + (1 if tot_frames % total_workers else 0)
                self.paths = paths[num_frame_per_worker * worker_idx:num_frame_per_worker * (worker_idx + 1)]

            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, 'empty folder'
            from PIL import Image
            tmp_img = Image.open(self.paths[0])
            self.width, self.height = tmp_img.size
        self.idx = 0

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        if self.input_type.startswith('video'):
            return self.get_frame_from_stream()
        else:
            return self.get_frame_from_list()

    def close(self):
        if self.input_type.startswith('video'):
            self.stream_reader.stdin.close()
            self.stream_reader.wait()

class Writer:
    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.')

        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 audio,
                                 video_save_path,
                                 pix_fmt='yuv420p',
                                 vcodec='libx264',
                                 loglevel='error',
                                 acodec='copy').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 video_save_path, pix_fmt='yuv420p', vcodec='libx264',
                                 loglevel='error').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))

    def write_frame(self, frame):
        frame = frame.astype(np.uint8).tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()

# ==============================================================================
#  3. MAIN INFERENCE LOGIC (Hybrid Flow/AI)
# ==============================================================================

def inference_video(args, video_save_path, device=None, total_workers=1, worker_idx=0):
    
    # --- SIMPLIFIED MODEL SELECTION ---
    # Map friendly names to actual Real-ESRGAN model names
    if args.mode == 'anime':
        args.model_name = 'realesr-animevideov3'
    elif args.mode == 'general':
        args.model_name = 'realesr-general-x4v3'
    # If using custom model_name via argument, keep it.
    
    # Set Process Interval based on Speed Preset
    if args.speed == 'slow':
        process_every = 0 # Every frame AI (Best Quality)
    elif args.speed == 'balanced':
        process_every = 2 # AI, Warp, AI, Warp (2x Speed)
    elif args.speed == 'fastest':
        process_every = 4 # AI, Warp, Warp, Warp, AI (4x Speed)
    else:
        process_every = args.process_every_manual

    # --- MODEL LOADING ---
    if args.model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    else:
        raise ValueError(f'Model {args.model_name} does not exist.')

    model_path = os.path.join('weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        device=device,
    )

    if args.face_enhance:
        if 'anime' in args.model_name:
            print('Face enhance is not supported/needed for anime models. Disabling.')
            args.face_enhance = False
        else:
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=args.outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler)
    else:
        face_enhancer = None

    # --- VIDEO PIPELINE START ---
    
    reader = Reader(args, total_workers, worker_idx)
    audio = reader.get_audio()
    height, width = reader.get_resolution()
    fps = reader.get_fps()
    writer = Writer(args, audio, height, width, video_save_path, fps)

    # State variables
    prev_lr_gray = None
    prev_hr_img = None
    frame_counter = 0
    stats = {'ai': 0, 'warp': 0}

    print(f"\nProcessing Config:")
    print(f"  Mode: {args.mode} ({args.model_name})")
    print(f"  Speed: {args.speed} (Process AI every {process_every} frames)")
    print(f"  Tiling: {args.tile}\n")

    pbar = tqdm(total=len(reader), unit='frame', desc='Upscaling')
    
    while True:
        img = reader.get_frame()
        if img is None:
            break

        curr_lr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        output = None
        mode = "ai"

        try:
            # 1. First Frame / Reset
            if prev_hr_img is None or prev_lr_gray is None:
                mode = "ai"
            
            # 2. Slow Mode (No Warping)
            elif process_every == 0:
                mode = "ai"

            else:
                # 3. Scene Cut Detection
                # If the scene changes drastically (>15 score), force AI reset
                diff_score = get_scene_change_score(prev_lr_gray, curr_lr_gray)
                if diff_score > 15.0:
                    mode = "ai"
                    frame_counter = 0 # Reset interval
                
                # 4. Keyframe Interval
                elif (frame_counter % process_every) == 0:
                    mode = "ai"
                
                # 5. Warping
                else:
                    mode = "warp"

            # --- EXECUTION ---
            if mode == "ai":
                if args.face_enhance:
                    _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    output, _ = upsampler.enhance(img, outscale=args.outscale)
                stats['ai'] += 1

            elif mode == "warp":
                # Calc flow (resized) -> Scale Flow -> Warp HR
                flow_lr = get_optical_flow(prev_lr_gray, curr_lr_gray)
                flow_hr_magnitude = flow_lr * args.outscale
                h_hr, w_hr = prev_hr_img.shape[:2]
                flow_hr_resized = cv2.resize(flow_hr_magnitude, (w_hr, h_hr), interpolation=cv2.INTER_LINEAR)
                output = warp_image(prev_hr_img, flow_hr_resized)
                stats['warp'] += 1

            writer.write_frame(output)
            
            # Update history
            prev_hr_img = output
            prev_lr_gray = curr_lr_gray
            frame_counter += 1

        except RuntimeError as error:
            print('Error:', error)
            print('Try reducing --tile size (e.g. -t 256)')
        
        torch.cuda.synchronize(device)
        pbar.update(1)
        pbar.set_postfix(stats)

    reader.close()
    writer.close()
    print(f"\nDone! Stats: {stats}")

def run(args):
    args.video_name = osp.splitext(os.path.basename(args.input))[0]
    video_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4')

    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
        os.makedirs(tmp_frames_folder, exist_ok=True)
        os.system(f'ffmpeg -i {args.input} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {tmp_frames_folder}/frame%08d.png')
        args.input = tmp_frames_folder

    num_gpus = torch.cuda.device_count()
    num_process = num_gpus * args.num_process_per_gpu
    
    # We only support single process for this optimized version currently 
    # (because temporal warping needs sequential frames)
    inference_video(args, video_save_path, torch.device(0), 1, 0)

    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
        shutil.rmtree(tmp_frames_folder)

def main():
    parser = argparse.ArgumentParser()
    
    # --- SIMPLIFIED ARGUMENTS ---
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input video, image or folder')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    
    # Easy Mode Selection
    parser.add_argument('--mode', type=str, default='anime', choices=['anime', 'general'], 
                        help='Simplified mode selection (anime or general video)')
    
    # Speed/Quality Selection
    parser.add_argument('--speed', type=str, default='balanced', choices=['slow', 'balanced', 'fastest'], 
                        help='slow=Best Quality(No Warp), balanced=2x Speed, fastest=4x Speed')
    
    parser.add_argument('-s', '--outscale', type=float, default=2, help='Upsampling scale (2 or 4)')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size (0=auto, 400=low VRAM)')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face (general mode only)')
    
    # --- ADVANCED/LEGACY ARGUMENTS ---
    parser.add_argument('--model_name', type=str, default='', help='(Advanced) Manually specify model name')
    parser.add_argument('--process_every_manual', type=int, default=0, help='(Advanced) Manual warp interval')
    parser.add_argument('--denoise_strength', type=float, default=0.5, help='Denoise strength for general model')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored video')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision')
    parser.add_argument('--fps', type=float, default=None, help='FPS of the output video')
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='The path to ffmpeg')
    parser.add_argument('--extract_frame_first', action='store_true')
    parser.add_argument('--num_process_per_gpu', type=int, default=1)
    parser.add_argument('--alpha_upsampler', type=str, default='realesrgan', help='The upsampler for the alpha channels')
    parser.add_argument('--ext', type=str, default='auto', help='Image extension')
    
    args = parser.parse_args()

    args.input = args.input.rstrip('/').rstrip('\\')
    os.makedirs(args.output, exist_ok=True)

    if mimetypes.guess_type(args.input)[0] is not None and mimetypes.guess_type(args.input)[0].startswith('video'):
        is_video = True
    else:
        is_video = False

    if is_video and args.input.endswith('.flv'):
        mp4_path = args.input.replace('.flv', '.mp4')
        os.system(f'ffmpeg -i {args.input} -codec copy {mp4_path}')
        args.input = mp4_path

    run(args)

if __name__ == '__main__':
    main()
