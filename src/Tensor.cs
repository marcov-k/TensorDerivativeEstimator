internal class Tensor
{
    public int Rank => Dimensions.Length;
    public readonly int[] Dimensions;
    public readonly double[] Data;
    public readonly int LinearSize;

    public Tensor(int[] dimensions)
    {
        Dimensions = dimensions;

        int linearSize = 1;
        foreach (int dim in dimensions) linearSize *= dim;
        LinearSize = linearSize;

        Data = new double[linearSize];
    }

    public static Tensor InitializeRandomTensor(int[] dimensions, double max, double min)
    {
        Tensor tensor = new(dimensions);

        for (int i = 0; i < tensor.LinearSize; ++i)
        {
            tensor.Data[i] = Random.Shared.NextDouble() * (max - min) + min;
        }

        return tensor;
    }

    // Tensor multiplication
    public static Tensor operator *(Tensor a, Tensor b)
    {
        if (a.Rank != b.Rank) throw new ArgumentException("Tensor ranks must match");

        int spatialSize = 1;
        for (int d = 0; d < a.Rank - 2; ++d)
        {
            if (a.Dimensions[d] != b.Dimensions[d]) throw new ArgumentException("Tensor spatial dimensions must match");
            spatialSize *= a.Dimensions[d];
        }

        if (a.Dimensions[^1] != b.Dimensions[^2]) throw new ArgumentException("Last 2 tensor dimensions must be a valid matrix multiplication pair");

        int[] resultDims = a.Dimensions;
        resultDims[^1] = b.Dimensions[^1];

        Tensor result = new(resultDims);

        int sharedDim = a.Dimensions[^1];
        int aMatrixSize = a.Dimensions[^2] * a.Dimensions[^1];
        int bMatrixSize = b.Dimensions[^2] * b.Dimensions[^1];
        int rMatrixSize = resultDims[^2] * resultDims[^1];

        for (int s = 0; s < spatialSize; ++s)
        {
            int aSpatialOffset = s * aMatrixSize;
            int bSpatialOffset = s * bMatrixSize;
            int rSpatialOffset = s * rMatrixSize;

            for (int i = 0; i < rMatrixSize; ++i)
            {
                int row = i / resultDims[^1];
                int col = i % resultDims[^1];

                for (int j = 0; j < sharedDim; ++j)
                {
                    result.Data[rSpatialOffset + i] += a.Data[aSpatialOffset + row * a.Dimensions[^1] + j] * b.Data[bSpatialOffset + j * b.Dimensions[^1] + col];
                }
            }
        }

        return result;
    }

    public static Tensor Transpose(Tensor t)
    {
        int[] transposeDims = t.Dimensions;
        (transposeDims[^2], transposeDims[^1]) = (transposeDims[^1], transposeDims[^2]);

        Tensor transpose = new(transposeDims);

        int spatialSize = 1;
        for (int d = 0; d < t.Rank - 2; ++d) spatialSize *= t.Dimensions[d];

        int matrixSize = t.Dimensions[^2] * t.Dimensions[^1];
        for (int s = 0; s < spatialSize; ++s)
        {
            int spatialOffset = s * matrixSize;

            for (int i = 0; i < matrixSize; ++i)
            {
                int originRow = i / t.Dimensions[^1];
                int originCol = i % t.Dimensions[^1];

                transpose.Data[spatialOffset + originCol * transposeDims[^1] + originRow] = t.Data[spatialOffset + originRow * t.Dimensions[^1] + originCol];
            }
        }

        return transpose;
    }
}
