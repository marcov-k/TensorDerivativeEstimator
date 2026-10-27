Console.WriteLine("Enter number of tensor dimensions: ");
int dimensionCount = int.Parse(Console.ReadLine()!);

Console.WriteLine("Enter number of parameters for function: ");
int parameterCount = int.Parse(Console.ReadLine()!);

DerivativeEstimator estimator = new(parameterCount, dimensionCount);

Console.WriteLine("Enter number of trials: ");
int trials = int.Parse(Console.ReadLine()!);

Console.WriteLine("\n\nEstimating derivatives...\n\n");
for (int t = 0; t < trials; ++t)
{
    Console.Write($"Trail {t + 1}: ");
    estimator.EstimateDerivative();
    Console.WriteLine();
}

Console.WriteLine("\n\nPress any key to close...");
Console.ReadKey();