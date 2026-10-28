using static Constants;

internal class DerivativeEstimator(int parameterCount, int dimensionCount)
{
    readonly int ParameterCount = parameterCount;
    readonly int DimensionCount = dimensionCount;

    public void EstimateDerivative()
    {
        Tensor input = GenerateInput();
        CompositeFunction function = GenerateFunction(input.Dimensions);

        var (partialInput, partialParameter) = ComputePartialDerivative(function, input);
        var (finiteInput, finiteParameter) = EstimateIndividualFiniteDifferenceDerivative(function, input);

        double inputError = 0.0;
        double parameterError = 0.0;
        for (int i = 0; i < input.LinearSize; ++i) inputError += Math.Abs(partialInput.Data[i] - finiteInput.Data[i]);
        for (int i = 0; i < partialParameter.LinearSize; ++i) parameterError += Math.Abs(partialParameter.Data[i] - finiteParameter.Data[i]);

        inputError /= input.LinearSize;
        parameterError /= partialParameter.LinearSize;

        Console.WriteLine($"Average input error: {inputError:N5} : Average parameter error: {parameterError:N5}");
    }

    static (Tensor input, Tensor parameter) ComputePartialDerivative(CompositeFunction function, Tensor input)
    {
        Tensor[] intermediates = function.Compute(input).intermediates;

        Tensor inputDeriv = new(intermediates[^1].Dimensions);
        for (int i = 0; i < inputDeriv.LinearSize; ++i) inputDeriv.Data[i] = 1.0;

        for (int p = function.Parameters.Length - 1; p > 0; --p)
        {
            inputDeriv *= Tensor.Transpose(function.Parameters[p]);
        }

        Tensor parameterDeriv = Tensor.Transpose(input) * inputDeriv;
        inputDeriv *= Tensor.Transpose(function.Parameters[0]);

        return (inputDeriv, parameterDeriv);
    }

    static (Tensor input, Tensor parameter) EstimateIndividualFiniteDifferenceDerivative(CompositeFunction function, Tensor input)
    {
        double baseResult = function.Compute(input).result;

        Tensor inputDeriv = new(input.Dimensions);
        Tensor parameterDeriv = new(function.Parameters[0].Dimensions);

        for (int i = 0; i < input.LinearSize; ++i)
        {
            double originalInput = input.Data[i];
            input.Data[i] += Epsilon;

            double newResult = function.Compute(input).result;
            inputDeriv.Data[i] = (newResult - baseResult) / Epsilon;

            input.Data[i] = originalInput;
        }

        for (int i = 0; i < function.Parameters[0].LinearSize; ++i)
        {
            double originalParam = function.Parameters[0].Data[i];
            function.Parameters[0].Data[i] += Epsilon;

            double newResult = function.Compute(input).result;
            parameterDeriv.Data[i] = (newResult - baseResult) / Epsilon;

            function.Parameters[0].Data[i] = originalParam;
        }

        return (inputDeriv, parameterDeriv);
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