using System;

namespace NN
{
    class Sigmoid : IActivation
    {
        public double activate(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public double derivative(double x)
        {
            return x * (1.0 - x);
        }
    }
}