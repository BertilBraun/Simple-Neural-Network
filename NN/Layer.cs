namespace NN
{
    abstract class Layer : BaseLayer
    {
        public BaseLayer Before { get; set; }
        public OutputLayer LastLayer { get; set; }
        public IActivation Activation { get; set; }

        public Layer(int neuronCount, BaseLayer before, IActivation activation) 
            : base(neuronCount)
        {
            Before = before;
            Activation = activation;
        }

        public void Forward()
        {
            foreach (Neuron neuron in Neurons)
            {
                double total = 0;

                for (int j = 0; j < Before.NeuronCount; j++)
                    total += (neuron.Weights[j].Value * Before.Neurons[j].Output);

                neuron.Value = total + neuron.Bias;
                neuron.Output = Activation.activate(neuron.Value);
            }
        }

        public void ApplyLearning(double learningRate)
        {
            for (int j = 0; j < NeuronCount; j++)
            {
                Neuron n = Neurons[j];

                for (int k = 0; k < n.Weights.Count; k++)
                    n.Weights[k].Value += (-learningRate * n.ErrorSignal * Before.Neurons[k].Output);
            }
        }

        public abstract void CalculateDelta();
    }
}
