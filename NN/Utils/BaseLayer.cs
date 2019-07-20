using System.Collections.Generic;

namespace NN
{
    abstract class BaseLayer
    {
        public List<Neuron> Neurons { get; set; }

        public int NeuronCount { get { return Neurons.Count; } }

        public BaseLayer(int neuronCount)
        {
            Neurons = new List<Neuron>();

            for (int i = 0; i < neuronCount; i++)
                Neurons.Add(new Neuron());
        }
    }
}
