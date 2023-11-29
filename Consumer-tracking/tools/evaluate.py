import argparse
import os
import os.path as osp
import time
import cv2
import torch
import motmetrics as mm
import glob
from collections import OrderedDict
from pathlib import Path

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.sort_tracker.sort import Sort
from yolox.deepsort_tracker.deepsort import DeepSort
from yolox.motdt_tracker.motdt_tracker import OnlineTracker
from yolox.strongsort_tracker.tracker import StrongSORT
from yolox.tracking_utils.timer import Timer

trackerTimer = Timer()
timer = Timer()

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Evaluation on Consumers Dataset")

    parser.add_argument("-b", "--benchmark", default=False, action="store_true", help="Benchmark: measure fps only")
    parser.add_argument("--path", "--data-path", help="Path to dataset", default='datasets/Consumers/')
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        required=True,
        type=str,
        help="Experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, required=True, type=str, help="Path to checkpoint file for evaluation")
    parser.add_argument(
        "-d",
        "--device",
        default="gpu",
        type=str,
        help="Device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=True,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument('--consumers', default=True, action='store_true', help='Use Consumers dataset')

    # lstm args
    parser.add_argument("--lstm", default=True, action='store_true')
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--lstm_neurons", type=int, default=128)
    parser.add_argument('--reset_hm', action='store_true')
    parser.add_argument('--reuse_hm', action='store_true')
    parser.add_argument('--not_set_cuda_env', action='store_true', help='used when training in slurm clusters')
    parser.add_argument('--lstm_weights', type=str, default='weights/lstm_128.pth', help='path to lstm weights file')

    # alg
    parser.add_argument("--method", default="byte", type=str, help="tracking method", choices=('byte','sort', 'deepsort', 'motdt', 'strongsort'))
    # for deepsort
    parser.add_argument("--reid", type=str, default='weights/ckpt.t7', help="reid model folder")
    # for motdt
    parser.add_argument("--googlenet", type=str, default='weights/googlenet_part8_all_xavier_ckpt_56.h5', help="reid model folder")

    parser.add_argument("--exp", "--experiment_name", type=str, default=None, help='Path to save experiment results')
    return parser


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


def eval_seqs(data_path, results_folder):
    print('Evaluating sequences...')

    mm.lap.default_solver = 'lap'
    gtfiles = glob.glob(osp.join(data_path, '*/gt/gt.txt'))
    tsfiles = [f for f in glob.glob(osp.join(results_folder, '*.txt')) if not osp.basename(f).startswith('eval')]

    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(osp.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
            'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
            'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)

    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])

    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                        'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']

    print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logger.info('Evaluation completed.')
    return



class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            device=torch.device("cpu"),
            fp16=False
    ):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        if img is None:
            raise ValueError("Empty image: ", img_info["file_name"])

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

        return outputs, img_info

def track_seq(predictor, vis_folder, args, seqpath):

    files = [osp.join(seqpath, f) for f in sorted(os.listdir(seqpath)) if f.endswith('g')]

    num_frames = len(files)

    if args.method == 'sort':
        tracker = Sort(args.track_thresh)
    elif args.method == 'deepsort':
        tracker = DeepSort(args.reid, min_confidence=args.track_thresh)
    elif args.method == 'motdt':
        tracker = OnlineTracker(args.googlenet, min_cls_score=args.track_thresh)
    elif args.method == 'strongsort':
        tracker = StrongSORT(args.reid)
    else:
        tracker = BYTETracker(args, frame_rate=args.fps)

    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            trackerTimer.tic()
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, img_path[27:])
            trackerTimer.toc()

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                if 'sort' in args.method:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                else:
                    tlwh = t.tlwh
                    tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    if 'sort' not in args.method:
                        online_scores.append(t.score)
                        results.append(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n")
                    else:
                        results.append(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},-1,-1,-1,-1\n")

        timer.toc()

        if frame_id % 20 == 0:
            logger.info('Processing frame {}/{} ({:.2f} fps)'.format(frame_id, num_frames, 1. / max(1e-5, timer.average_time)))

    if not args.benchmark:
        res_file = osp.join(vis_folder, args.name + ".txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"Saving results to {res_file}.")


def main(exp, args, seqpath):
    if args.experiment_name is None:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.lstm == True:
        vis_folder = osp.join(output_dir, f'track_results_{args.method}_lstm')
    else:
        vis_folder = osp.join(output_dir, f'track_results_{args.method}')
    os.makedirs(vis_folder, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    #logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if args.ckpt is None:
        logger.error("No checkpoint provided.")
        raise 
    else:
        ckpt_file = args.ckpt
    logger.info("Loading checkpoint...")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info(f"Loaded checkpoint {ckpt_file} successfully.")

    if args.fuse:
        #logger.info("Fusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()

    predictor = Predictor(model, exp, args.device, args.fp16)

    track_seq(predictor, vis_folder, args, seqpath)

if __name__ == "__main__":
    args = make_parser().parse_args()

    logger.info("Args: {}".format(args))

    data_path = args.path
    fp16 = args.fp16
    device = args.device

    test_seqs = [
            'Sequence_ID121',
            'Sequence_ID122',
            'Sequence_ID123',
            'Sequence_ID124',
            'Sequence_ID125',
            'Sequence_ID126',
            'Sequence_ID127',
            'Sequence_ID128',
            'Sequence_ID129',
            'Sequence_ID130',
            'Sequence_ID131',
            'Sequence_ID132',
            'Sequence_ID133',
            'Sequence_ID134',
            'Sequence_ID135',
            'Sequence_ID136',
            'Sequence_ID137',
            'Sequence_ID138',
            'Sequence_ID139',
            'Sequence_ID140',
            'Sequence_ID141',
            'Sequence_ID142',
            'Sequence_ID143',
            'Sequence_ID144',
            'Sequence_ID145',
            ]

    mainTimer = Timer()
    mainTimer.tic()

    for seq in test_seqs:
        args.name = seq
        args.fp16 = fp16
        args.device = device
        seqpath = osp.join(data_path, seq)
        exp = get_exp(args.exp_file, args.name)
        main(exp, args, seqpath)

    mainTimer.toc()
    print('Total elapsed time: ', mainTimer.total_time)
    print('Total time (detection + tracking): ' + str(timer.total_time) + ", FPS: " + str(1.0 /timer.average_time))
    print('Total time (tracking only): ' + str(trackerTimer.total_time) + ", FPS: " + str(1.0 / trackerTimer.average_time))

    if not args.benchmark:
        if args.lstm == True:
            results_folder = osp.join('YOLOX_outputs', args.experiment_name, f'track_results_{args.method}_lstm')
        else:
            results_folder = osp.join('YOLOX_outputs', args.experiment_name, f'track_results_{args.method}')
        eval_seqs(args.path, results_folder)
