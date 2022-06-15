def transform_to_new_range(x, minimum, maximum, scale=10):
    x -= minimum
    x /= (maximum - minimum)
    x *= scale

    return x