namespace NN
{
    interface IActivation
    {
        double activate(double x);
        double derivative(double x);
    }
}
