import argparse


def bool_parse(arg):
    if arg.lower() in ("true", "t", "yes", "y", "1"):
        return True
    elif arg.lower() in ("false", "f", "no", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def groupby_legal_precision_annotation(legal_precision_annotation):
    if legal_precision_annotation == 1:
        group = 0  # "off-track"
    elif legal_precision_annotation <= 5:
        # By default, it will be > 1 since the first if capture it
        group = 1  # "seems imprecise"
    else:
        # Same logic that the previous group, namely greater than 5
        group = 2  # "accurate"
    return group
