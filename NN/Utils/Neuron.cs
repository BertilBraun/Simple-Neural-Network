using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN
{
    class Weight
    {
        public double Value { get; set; }

        public Weight()
        {
            Random rand = new Random();

            Value = rand.NextDouble();
        }
    }

    class Neuron
    {
        public List<Weight> Weights { get; set; }
        public double Bias { get; set; }
        public double ErrorSignal { get; set; }
        public double Value { get; set; }
        public double Output { get; set; }

        public Neuron()
        {
            Random rand = new Random();

            Bias = rand.NextDouble();
            ErrorSignal = 0;
            Output = 0;

            Weights = new List<Weight>();
        }

    }
}
