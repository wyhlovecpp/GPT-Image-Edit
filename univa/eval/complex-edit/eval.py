import argparse
from glob import glob
import json
from loguru import logger
import os
import os.path as osp

from datasets import load_dataset

from complex_edit.utils import dict_mean, setup_logger
from complex_edit.eval import AlignmentEvaluator, QualityEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image editing model.")
    parser.add_argument("--path", "-p", type=str, required=True,
                        help="Path to the directory containing the output images.")
    parser.add_argument("--complexity", "-c", type=int, choices=list(range(1, 9)), required=True,
                        help="Complexity level")
    parser.add_argument("--image-type", type=str, choices=["real", "syn"], default="real",
                        help="input image type")
    parser.add_argument("-n", type=int, default=20,
                        help="Total number of measurements for one sample.")
    parser.add_argument(
        "-m", type=int, default=5,
        help="Maximum number of responses per call. e.g. n = 10 and m = 5, then 2 calls will be made."
    )
    parser.add_argument("--num-processes", type=int, default=16, help="Number of processes.")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume instead of replacing.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    output_image_paths = sorted(glob(osp.join(args.path, "*.png")))

    dataset = load_dataset("UCSC-VLAA/Complex-Edit")

    input_images = dataset[f"test_{args.image_type}"]["image"]
    instructions = [
        edit["compound"][args.complexity - 1]["compound_instruction"]
        for edit in dataset[f"test_{args.image_type}"]["edit"]
    ]

    assert len(output_image_paths) == len(dataset[f"test_{args.image_type}"]), \
        f"Number of output images does not match the number of samples: {len(output_image_paths)}"

    alignment_evaluator = AlignmentEvaluator(
        if_rubric=True,
        if_cot=True,
        if_resume=args.resume,
        n=args.n, m=args.m,
        num_processes=args.num_processes
    )
    quality_evaluator = QualityEvaluator(
        if_rubric=True,
        if_cot=False,
        if_resume=args.resume,
        n=args.n, m=args.m,
        num_processes=args.num_processes
    )

    save_eval_paths = [
        osp.join(
            args.path,
            alignment_evaluator.result_folder_name,
            f"{osp.basename(p).split('.')[0]}.json"
        )
        for p in output_image_paths
    ]
    os.makedirs(osp.join(args.path, alignment_evaluator.result_folder_name), exist_ok=True)
    setup_logger(output=osp.join(args.path, alignment_evaluator.result_folder_name, "log.txt"))

    logger.info(f"Evaluating Alignment for images {args.path}.")
    alignment_results = alignment_evaluator.eval(
        input_images=input_images,
        output_images=output_image_paths,
        instructions=instructions,
        save_paths=save_eval_paths,
    )

    final_alignment = dict_mean(alignment_results)
    json.dump(
        final_alignment,
        open(
            osp.join(args.path, alignment_evaluator.result_folder_name, "final_result.json"),
            "w"
        ),
        indent=4
    )
    logger.info(f"Final alignment: {final_alignment}")

    save_eval_paths = [
        osp.join(
            args.path,
            quality_evaluator.result_folder_name,
            f"{osp.basename(p).split('.')[0]}.json"
        )
        for p in output_image_paths
    ]
    os.makedirs(osp.join(args.path, quality_evaluator.result_folder_name), exist_ok=True)
    setup_logger(output=osp.join(args.path, quality_evaluator.result_folder_name, "log.txt"))

    logger.info(f"Evaluating Quality for images {args.path}.")
    quality_results = quality_evaluator.eval(
        output_images=output_image_paths,
        instructions=instructions,
        save_paths=save_eval_paths,
    )

    final_quality = dict_mean(quality_results)
    json.dump(
        final_quality,
        open(
            osp.join(args.path, quality_evaluator.result_folder_name, "final_result.json"),
            "w"
        ),
        indent=4
    )
    logger.info(f"Final quality: {final_quality}")

    os.makedirs(osp.join(args.path, "overall"), exist_ok=True)
    setup_logger(output=osp.join(args.path, "overall", "log.txt"))
    overall_results = []
    for alignment_result, quality_result, p, instruction in zip(
        alignment_results, quality_results, output_image_paths, instructions
    ):
        overall = {}
        overall.update(alignment_result)
        overall.update(quality_result)
        overall["overall"] = sum(overall.values()) / len(overall)
        overall["instruction"] = instruction
        json.dump(
            overall,
            open(osp.join(args.path, "overall", f"{osp.basename(p).split('.')[0]}.json"), "w"),
            indent=4
        )

        overall.pop("instruction")
        overall_results.append(overall)

    final_overall = dict_mean(overall_results)
    json.dump(
        final_quality,
        open(osp.join(args.path, "overall", "final_result.json"), "w"),
        indent=4
    )
    logger.info(f"Final overall: {final_overall}")
