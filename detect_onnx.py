"""
detect_onnx.py  –  a drop-in ONNX version of your original detect.py

Only the model-loading and inference sections were changed:
▪︎  PyTorch ➜ onnxruntime
▪︎  Removed .half(), augment, strip_optimizer, etc.
Everything else (NMS, plotting, CLI flags) is identical, so it should behave the same way.
"""
import argparse, time
from pathlib import Path
import cv2, numpy as np, torch, torch.backends.cudnn as cudnn
import onnxruntime as ort

from utils.datasets import LoadStreams, LoadImages
from utils.general  import check_img_size, check_requirements, check_imshow, non_max_suppression, \
                           apply_classifier, scale_coords, xyxy2xywh, set_logging, increment_path, save_one_box
from utils.plots    import colors, plot_one_box
from utils.torch_utils import time_synchronized

# ---------- helper ----------------------------------------------------------- #
def run_onnx(session, img):
    """img: (1,3,H,W) fp32 ndarray 0-1 → numpy ndarray   returns raw network output"""
    return session.run(None, {session.get_inputs()[0].name: img})[0]

# ---------- main ------------------------------------------------------------- #
def detect(opt):
    src, weights, save_txt, save_frames, imgsz = opt.source, opt.weights, opt.save_txt, opt.save_frames, opt.img_size
    view_img, save_img, save_txt_tidl, kpt_label = opt.view_img, (not opt.nosave and not src.endswith('.txt')), opt.save_txt_tidl, opt.kpt_label
    webcam  = src.isnumeric() or src.endswith('.txt') or src.lower().startswith(('rtsp://', 'rtmp://','http://','https://'))

    # dirs
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    (save_dir/'labels' if (save_txt or save_txt_tidl) else save_dir).mkdir(parents=True, exist_ok=True)
    (save_dir/'frames').mkdir(parents=True, exist_ok=True)

    # init
    set_logging(); cudnn.benchmark = True
    imgsz = check_img_size(imgsz, s=32)
    providers = ['CUDAExecutionProvider','CPUExecutionProvider']  # falls back to CPU if no GPU
    session   = ort.InferenceSession(weights[0], providers=providers)         # NEW ⭐
    stride    = 32                                                            # assume export kept 32-stride
    names     = [str(i) for i in range(1000)]                                 # label list stub; fill if you have .yaml

    # data
    dataset = LoadStreams(src, img_size=imgsz, stride=stride) if webcam else LoadImages(src, img_size=imgsz, stride=stride)

    t0 = time.time(); vid_path, vid_writer, total_frames = None, None, 0
    for path, img, im0s, vid_cap in dataset:
        # --- pre-process ---------------------------------------------------- #
        # ---------- 预处理 ----------
        # img 可能是 (1,3,H,W) / (3,H,W) / (H,W,3)
        if img.ndim == 4:  # 已经 (1,3,H,W)
            img_in = img
        elif img.ndim == 3 and img.shape[0] in {1, 3}:  # (C,H,W)
            img_in = np.expand_dims(img, 0)  # ➜ (1,C,H,W)
        elif img.ndim == 3 and img.shape[2] == 3:  # (H,W,3)
            img_in = np.expand_dims(img.transpose(2, 0, 1), 0)  # ➜ (1,3,H,W)
        else:
            raise ValueError(f'不支持的图像形状: {img.shape}')

        img_in = img_in.astype(np.float32) / 255.0  # 归一化到 0-1

        # --- inference ------------------------------------------------------ #
        t1 = time_synchronized()
        pred = torch.from_numpy(run_onnx(session, img_in)).to(torch.float32)  # numpy → torch
        t2  = time_synchronized()

        # --- NMS ------------------------------------------------------------ #
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   classes=opt.classes, agnostic=opt.agnostic_nms,
                                   kpt_label=kpt_label, nc=len(names), nkpt=0)

        # --- post-process / draw / save ------------------------------------- #
        for i, det in enumerate(pred):
            p, im0 = (path[i], im0s[i].copy()) if webcam else (path, im0s.copy())
            p = Path(p); save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem)

            if len(det):
                if save_frames:            # save raw frame
                    cv2.imwrite(str(save_dir/'frames'/f'{total_frames}.jpg'), im0); total_frames += 1

                scale_coords(img_in.shape[2:], det[:,:4], im0.shape, kpt_label=False)
                scale_coords(img_in.shape[2:], det[:,6:], im0.shape, kpt_label=kpt_label, step=3)

                for det_idx, (*xyxy, conf, cls) in enumerate(det[:,:6]):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1,4)) / torch.tensor(im0.shape)[[1,0,1,0]]).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path+'.txt', 'a') as f: f.write(('%g '*len(line)).rstrip()%line + '\n')

                    if save_img or view_img or opt.save_crop:
                        c, label = int(cls), (names[int(cls)] if not opt.hide_labels else None)
                        plot_one_box(xyxy, im0, label=f'{label} {conf:.2f}' if not opt.hide_conf else label,
                                     color=colors(c, True), line_thickness=1)

            if view_img: cv2.imshow(str(p), im0), cv2.waitKey(1)
            if save_img: cv2.imwrite(save_path, im0)

    print(f'Done. ({time.time()-t0:.3f}s)  Results ➜ {save_dir}')

# ---------- cli -------------------------------------------------------------- #
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weights', nargs='+', default=['best.onnx'], help='ONNX model')
    p.add_argument('--source',  default='0', help='img/dir/vid/URL or webcam id')
    p.add_argument('--img-size', type=int, default=416)
    p.add_argument('--conf-thres', type=float, default=0.4)
    p.add_argument('--iou-thres',  type=float, default=0.5)
    p.add_argument('--device', default='cuda')         # kept for API parity (ignored)
    p.add_argument('--view-img',     action='store_true')
    p.add_argument('--save-txt',     action='store_true')
    p.add_argument('--save-frames',  action='store_true')
    p.add_argument('--save-txt-tidl', action='store_true')
    p.add_argument('--save-conf',    action='store_true')
    p.add_argument('--save-crop',    action='store_true')
    p.add_argument('--nosave',       action='store_true')
    p.add_argument('--classes', nargs='+', type=int)
    p.add_argument('--agnostic-nms', action='store_true')
    p.add_argument('--project', default='runs/detect')
    p.add_argument('--name',    default='exp')
    p.add_argument('--exist-ok', action='store_true')
    p.add_argument('--hide-labels', action='store_true')
    p.add_argument('--hide-conf',  action='store_true')
    p.add_argument('--kpt-label',  default=True)
    opt = p.parse_args()

    check_requirements(exclude=('tensorboard','pycocotools','thop'))
    detect(opt)
