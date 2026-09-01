"""Command-line options for PGMAN training and evaluation."""

import argparse


def parse_opts(argv=None):
    parser = argparse.ArgumentParser(
        description="Train PGMAN for multimodal micro-video emotion recognition"
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument("--root_path", required=True, help="Dataset/project root")
    paths.add_argument("--video_path", default="MeiTu/video")
    paths.add_argument("--audio_path", default="MeiTu/audio")
    paths.add_argument("--text_path", default="annotations/mtsvrc_title.json")
    paths.add_argument("--annotation_path", default="annotations/mtsvrc_title.json")
    paths.add_argument("--fold_csv", default="annotations/mtsvrc.csv")
    paths.add_argument("--result_paths", default="results/main")
    paths.add_argument("--expr_name", default="")
    paths.add_argument("--pretrained", default="", help="Training checkpoint")
    paths.add_argument("--visual_pretrained", default="", help="TimeSformer checkpoint")

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", default="ME5", choices=("ME5", "ek6"))
    data.add_argument("--batch_size", default=8, type=int)
    data.add_argument("--sample_size", default=224, type=int)
    data.add_argument("--n_frames", default=8, type=int)
    data.add_argument("--fps", default=30, type=int)
    data.add_argument("--n_threads", default=8, type=int)
    data.add_argument("--drop_last", dest="dl", action="store_true")

    model = parser.add_argument_group("model")
    model.add_argument("--n_classes", default=5, type=int)
    model.add_argument("--audio_embed_size", default=256, type=int)
    model.add_argument("--audio_n_segments", default=8, type=int)
    model.add_argument("--need_audio", action=argparse.BooleanOptionalAction, default=True)
    model.add_argument("--need_text", action=argparse.BooleanOptionalAction, default=True)

    training = parser.add_argument_group("training")
    training.add_argument("--mode", default="main", choices=("pretrain", "main"))
    training.add_argument("--loss_func", default="ce", choices=("ce", "pcce_ve8"))
    training.add_argument("--lambda_0", default=0.5, type=float)
    training.add_argument("--learning_rate", default=1e-5, type=float)
    training.add_argument("--weight_decay", default=1e-4, type=float)
    training.add_argument("--n_epochs", default=200, type=int)
    training.add_argument("--seed", default=99, type=int)
    training.add_argument("--debug", action="store_true")

    return parser.parse_args(argv)
