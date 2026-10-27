using static Constants;

internal class CompositeFunction
{
    public readonly Tensor[] Parameters;

    public CompositeFunction(int parameterCount, int[] inputDimensions)
    {
        Parameters = new Tensor[parameterCount];

        int[] paramDims = [..inputDimensions];
        for (int p = 0; p < parameterCount; ++p)
        {
            paramDims[^2] = paramDims[^1];
            paramDims[^1] = Random.Shared.Next(MinDimensionLength, MaxDimensionLength + 1);

            Parameters[p] = Tensor.InitializeRandomTensor(paramDims, MaxParameterValue, MinParameterValue);
        }
    }

    public (double result, Tensor[] intermediates) Compute(Tensor input)
    {
        Tensor result = input.Copy();
        var intermediates = new Tensor[Parameters.Length];

        for (int p = 0; p < Parameters.Length; ++p)
        {
            result *= Parameters[p];
            intermediates[p] = result.Copy();
        }

        return (Tensor.Sum(result), intermediates);
    }
}
