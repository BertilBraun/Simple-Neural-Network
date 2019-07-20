namespace NN
{
    class InputLayer : BaseLayer, ILayerAfter
    {
        public BaseLayer After { get; set; }

        public InputLayer(int neuronCount)
            : base(neuronCount)
        { }
    }
}
