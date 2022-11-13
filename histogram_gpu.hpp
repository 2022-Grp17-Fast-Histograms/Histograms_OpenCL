int adjustDimension(int dimension, int blockDimension) {
    return dimension - (dimension % blockDimension);
}