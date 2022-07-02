import argparse

from logic.patcher import Patcher


def main(params):

    patcher = Patcher(
        load_directory=params.load_directory,
        save_directory=params.save_directory,
        file_type=params.file_type,
        centres_per_dimension=params.centres_per_dimension,
        perfect_truth=params.perfect_truth,
        patch_size=params.patch_size,
        scale_dist=params.scale_dist,
        offset_multiplier=params.offset_multiplier,
        rescale=params.rescale,
        save_type=params.save_type,
    )

    patcher.create_and_save_all_patches_and_labels()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-ld", "--load_directory", default="/Users/fryderykkogl/Data/patches/val_nii_small",
                        help="Directory containing all niftis")
    parser.add_argument("-sd", "--save_directory", default="/Users/fryderykkogl/Data/patches/val_npy",
                        help="Directory to save the numpy patches")
    parser.add_argument("-ft", "--file_type", default="nii.gz", choices=["nii", "nii.gz"],
                        help="File type of the niftis")
    parser.add_argument("-cpd", "--centres_per_dimension", default=6,
                        help="Amount of centres per dimension")
    parser.add_argument("-pt", "--perfect_truth", default=False,
                        help="If true, the patches are extracted from the same volumes, otherwise from pairs")
    parser.add_argument("-ps", "--patch_size", default=32,
                        help="Size of the patches")
    parser.add_argument("-sdst", "--scale_dist", default=1.5,
                        help="Should be bigger than 1. If it is one, then the unrelated patch is adjacent to the "
                             "normal patch")
    parser.add_argument("-om", "--offset_multiplier", default=4,
                        help="Standard offset is [0,0,1] and this will be multiplied by this argument")
    parser.add_argument("-rs", "--rescale", default=True,
                        help="If true, the volumes are divided by their maximum value")
    parser.add_argument("-st", "--save_type", default="float16", choices=["uint8", "float16"])

    args = parser.parse_args()

    main(args)
