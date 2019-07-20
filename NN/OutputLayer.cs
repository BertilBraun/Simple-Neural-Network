namespace NN
{
    class OutputLayer : Layer
    {
        public double[] CorrectOutput { get; set; }
        public double[] Output
        {
            get
            {
                double[] output = new double[NeuronCount];
                for (int i = 0; i < NeuronCount; i++)
                    output[i] = Neurons[i].Output;
                return output;
            }
        }

        public OutputLayer(int neuronCount, BaseLayer before, IActivation activation)
            : base(neuronCount, before, activation)
        {
        }

        public override void CalculateDelta()
        {
            for (int i = 0; i < NeuronCount; i++)
            {
                Neuron neuron = Neurons[i];

                neuron.ErrorSignal = Activation.derivative(neuron.Value) * (neuron.Output - CorrectOutput[i]);
            }
        }
    }
}
