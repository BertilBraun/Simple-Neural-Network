namespace NN
{
    class HiddenLayer : Layer, ILayerAfter
    {
        public BaseLayer After { get; set; }

        public HiddenLayer(int neuronCount, BaseLayer before, IActivation activation) 
            : base(neuronCount, before, activation)
        { }

        public override void CalculateDelta()
        {
            for (int k = 0; k < NeuronCount; k++)
            {
                double sum = 0;
                for (int i = 0; i < After.NeuronCount; i++)
                    sum += After.Neurons[i].ErrorSignal * After.Neurons[i].Weights[k].Value;

                Neurons[k].ErrorSignal = Activation.derivative(Neurons[k].Value) * sum;
            }
        }
    }
}
