Tensor a = new([2, 2]);
Tensor b = new([2, 2]);
for (int i = 0; i < a.LinearSize; ++i)
{
    a.Data[i] = i;
    b.Data[i] = i * i;
}

var result = a * b;

Console.WriteLine("Contents of tensor a: ");
for (int row = 0; row < 2; ++row)
{
    for (int col = 0; col < 2; ++col)
    {
        Console.Write($"{a.Data[row * 2 + col]} ");
    }
    Console.WriteLine();
}

Console.WriteLine("\nContents of tensor b: ");
for (int row = 0; row < 2; ++row)
{
    for (int col = 0; col < 2; ++col)
    {
        Console.Write($"{b.Data[row * 2 + col]} ");
    }
    Console.WriteLine();
}

Console.WriteLine("\nContents of result tensor: ");
for (int row = 0; row < 2; ++row)
{
    for (int col = 0; col < 2; ++col)
    {
        Console.Write($"{result.Data[row * 2 + col]} ");
    }
    Console.WriteLine();
}

var resultTranspose = Tensor.Transpose(result);

Console.WriteLine("\nContents of transposed result tensor: ");
for (int row = 0; row < 2; ++row)
{
    for (int col = 0; col < 2; ++col)
    {
        Console.Write($"{resultTranspose.Data[row * 2 + col]} ");
    }
    Console.WriteLine();
}

Console.WriteLine("\n\nPress any key to close...");
Console.ReadKey();