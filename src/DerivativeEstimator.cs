using static Constants;

internal class DerivativeEstimator(int parameterCount, int dimensionCount)
{
    readonly int ParameterCount = parameterCount;
    readonly int DimensionCount = dimensionCount;

    public void EstimateDerivative()
    {
        Console.WriteLine("\n\nEstimating derivative...\n\n");
        Tensor input = GenerateInput();
        CompositeFunction function = GenerateFunction(input.Dimensions);

        Tensor partial = ComputePartialDerivative(function, input);
        Tensor finite = EstimateIndividualFiniteDifferenceDerivative(function, input);

        double error = 0.0;
        for (int i = 0; i < input.LinearSize; ++i) error += Math.Abs(partial.Data[i] - finite.Data[i]);

        error /= input.LinearSize;

        Console.Write("Input tensor dimensions: ");
        for (int i = 0; i < input.Dimensions.Length - 1; ++i) Console.Write($"{input.Dimensions[i]}, ");
        Console.WriteLine(input.Dimensions[^1]);
        Console.WriteLine($"Linear size: {input.LinearSize}");

        int totalParamValues = 0;
        Console.WriteLine("\nParameter tensor dimensions:");
        for (int p = 0; p < function.Parameters.Length; ++p)
        {
            var param = function.Parameters[p];
            totalParamValues += param.LinearSize;

            for (int i = 0; i < param.Dimensions.Length - 1; ++i) Console.Write($"{param.Dimensions[i]}, ");
            Console.WriteLine(param.Dimensions[^1]);
            Console.WriteLine($"Linear size: {param.LinearSize}");
        }

        Console.WriteLine($"\nTotal parameter values: {totalParamValues}");
        Console.WriteLine($"Total values involved: {totalParamValues + input.LinearSize}");

        Console.WriteLine($"\nAverage error between partial derivative and finite difference estimate: {error}");
    }

    static Tensor ComputePartialDerivative(CompositeFunction function, Tensor input)
    {
        Tensor[] intermediates = function.Compute(input).intermediates;

        Tensor derivative = new(intermediates[^1].Dimensions);
        for (int i = 0; i < derivative.LinearSize; ++i) derivative.Data[i] = 1.0;

        for (int p = function.Parameters.Length - 1; p >= 0; --p)
        {
            derivative *= Tensor.Transpose(function.Parameters[p]);
        }

        return derivative;
    }

    static Tensor EstimateIndividualFiniteDifferenceDerivative(CompositeFunction function, Tensor input)
    {
        double baseResult = function.Compute(input).result;

        Tensor derivative = new(input.Dimensions);

        for (int i = 0; i < input.LinearSize; ++i)
        {
            double originalInput = input.Data[i];
            input.Data[i] += Epsilon;

            double newResult = function.Compute(input).result;
            derivative.Data[i] = (newResult - baseResult) / Epsilon;

            input.Data[i] = originalInput;
        }

        return derivative;
    }

    Tensor GenerateInput()
    {
        var inputDims = new int[DimensionCount];
        for (int d = 0; d < DimensionCount; ++d)
        {
            inputDims[d] = Random.Shared.Next(MinDimensionLength, MaxDimensionLength + 1);
        }

        return Tensor.InitializeRandomTensor(inputDims, MaxParameterValue, MinParameterValue);
    }

    CompositeFunction GenerateFunction(int[] inputDimensions)
    {
        return new(ParameterCount, inputDimensions);
    }
}