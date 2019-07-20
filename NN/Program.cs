using System;

namespace NN
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork();

            nn.AddLayer(2, new Sigmoid());
            nn.AddLayer(3, new Sigmoid());
            nn.AddLayer(3, new Sigmoid());
            nn.AddLayer(1, new Sigmoid());

            nn.Build();


            double[][] input = new double[][] {
                new double[] { 0, 0 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 1, 1 }
            };

            double[][] and = new double[][] {
                new double[] { 0 },
                new double[] { 0 },
                new double[] { 0 },
                new double[] { 1 },
            };

            double[][] or = new double[][] {
                new double[] { 0 },
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 1 },
            };

            double[][] xor = new double[][] {
                new double[] { 0 },
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 0 },
            };

            nn.Train(input, xor, 10000, 0.25);

            nn.Predict(input[0]);
            nn.Predict(input[1]);
            nn.Predict(input[2]);
            nn.Predict(input[3]);

            Console.ReadLine();
        }
    }
}
