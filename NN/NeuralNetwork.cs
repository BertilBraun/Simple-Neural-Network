using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace NN
{
    class NeuralNetwork
    {
        public InputLayer InputLayer { get { return (InputLayer)layers[0]; } set { layers[0] = value; } }
        public OutputLayer OutputLayer { get { return (OutputLayer)layers[layers.Count - 1]; } set { layers[layers.Count - 1] = value; } }

        List<BaseLayer> layers = new List<BaseLayer>();

        public NeuralNetwork()
        {
        }

        public void AddLayer(int numberOfNeurons, IActivation activation)
        {
            if (layers.Count == 0)
                layers.Add(new InputLayer(numberOfNeurons));
            else
                layers.Add(new HiddenLayer(numberOfNeurons, layers[layers.Count - 1], activation));
        }

        public void Build()
        {
            var l = (HiddenLayer)layers[layers.Count - 1];
            layers[layers.Count - 1] = new OutputLayer(l.NeuronCount, l.Before, l.Activation);
            
            for (int i = 0; i < layers.Count; i++)
            {
                if (typeof(ILayerAfter).IsAssignableFrom(layers[i].GetType()))
                    ((ILayerAfter)layers[i]).After = layers[i + 1];

                if (typeof(Layer).IsAssignableFrom(layers[i].GetType()))
                {
                    var h = (Layer)layers[i];

                    h.LastLayer = OutputLayer;

                    for (int j = 0; j < h.NeuronCount; j++)
                        for (int k = 0; k < h.Before.NeuronCount; k++)
                            h.Neurons[j].Weights.Add(new Weight());
                }
            }
        }

        public void Forward(double[] inputs)
        {
            Debug.Assert(inputs.Length == InputLayer.NeuronCount, "Inputs don't match the input Neurons");

            for (int i = 0; i < inputs.Length; i++)
                InputLayer.Neurons[i].Output = inputs[i];

            for (int i = 1; i < layers.Count; i++)
                ((Layer)layers[i]).Forward();
        }

        public void Train(double[][] inputs, double[][] outputs, int epochs, double learningRate)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int index = 0; index < inputs.Length; index++)
                    Train(inputs[index], outputs[index], learningRate);

                if (epoch % (epochs / 10) == 0)
                    Console.WriteLine("In epoch " + epoch + " the NN evaluates to a cost of: " + evaluate(outputs[outputs.Length - 1]));
            }
        }

        void Train(double[] inputs, double[] outputs, double learningRate)
        {
            Debug.Assert(outputs.Length == OutputLayer.NeuronCount, "Outputs don't match the output Neurons");

            Forward(inputs);
            OutputLayer.CorrectOutput = outputs;

            for (int i = layers.Count - 1; i > 0; i--)
                ((Layer)layers[i]).CalculateDelta();

            for (int i = 1; i < layers.Count; i++)
                ((Layer)layers[i]).ApplyLearning(learningRate);
        }

        public double[] Predict(double[] inputs)
        {
            Forward(inputs);
            
            PrintPrediction(inputs, OutputLayer.Output);

            return OutputLayer.Output;
        }

        public void PrintPrediction(double[] inputs, double[] outputs)
        {
            Console.Write("With (");

            for (int i = 0; i < inputs.Length; i++)
                Console.Write(inputs[i].ToString() + (i != inputs.Length - 1 ? ", " : ""));

            Console.Write(") the predicted value is: [");

            for (int i = 0; i < outputs.Length; i++)
                Console.Write(outputs[i].ToString() + (i != outputs.Length - 1 ? ", " : ""));

            Console.WriteLine("]");
        }

        public double evaluate(double[] correct)
        {
            double[] output = OutputLayer.Output;
            double total = 0;

            for (int i = 0; i < correct.Length; i++)
                total += Math.Abs(output[i] - correct[i]);

            return total;
        }
    }
}
