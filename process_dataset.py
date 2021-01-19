import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Parse the argument to define processing parameters"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Source directory from which the data will be preprocessed",
    )
    parser.add_argument("--weight", type=str, help="model.pt path(s)")
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--project", default="runs/detect", help="save results to project/name"
    )

    args = parser.parse_args()
    print(args.__dict__)
    path_to_data = args.source_dir

    print("Start processing {}".format(path_to_data))
    for animal_folder in os.listdir(path_to_data):
        print("Processing {}".format(animal_folder))
        p = subprocess.Popen(
            [
                "python",
                "detect.py",
                "--source",
                os.path.join(path_to_data, animal_folder),
                "--img-size",
                str(args.img_size),
                "--weights",
                str(args.weight),
                "--name",
                str(animal_folder),
                "--conf-thres",
                str(args.conf_thres),
                "--crop",
                "--device",
                str(args.device),
            ]
        )
        p.wait()


if __name__ == "__main__":
    main()
