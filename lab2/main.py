import sys
from src import (
    load_image,
    binarize,
    get_contours_and_hull,
    get_tips_and_valleys,
    get_tip_valley_sequence,
    save_image,
)


def main(infile: str, outfile: str = "output.png"):
    source = load_image(infile)
    binarized = binarize(source)
    contour, hull = get_contours_and_hull(binarized)
    tips, valleys = get_tips_and_valleys(contour, hull)
    dots = get_tip_valley_sequence(tips, valleys)
    save_image(source, dots, outfile)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py infile.png [outfile.png]")
    else:
        main(*sys.argv[1:])
