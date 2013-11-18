package com.neuralnetwork.xor;

public class XORBackPropagationNetwork implements INeuralNetwork
{
    LayorInfo[] aLayers;
    INeuralNetwork[] aPreviousLayors;

    protected static float alpha = 0.5f;
    protected static float eta = 0.5f;

    IActivationFunction.IDifferentiableFunction phi;

    public XORBackPropagationNetwork()
    {
        phi = new IActivationFunction.SigmoidFunction(0.5f);

        INeuralNetwork firstPass = new SingleLayorNeuralNetwork();
        INeuralNetwork secondPass = new SingleLayorNeuralNetwork();

        ((SingleLayorNeuralNetwork)firstPass)
                .setNeurons(new Neuron(phi, 1f,1f,-1.5f),
                            new Neuron(phi, 1f,1f,-0.5f));

        ((SingleLayorNeuralNetwork)secondPass)
                .setNeurons(new Neuron(phi, -2f,1f,-0.5f));

        aLayers = new LayorInfo[2];

        aLayers[0] = new LayorInfo(firstPass);
        aLayers[1] = new LayorInfo(secondPass);
    }

    protected class LayorInfo
    {
        final public INeuralNetwork network;
        public NVector rawoutput;
        public NVector output;
        public NVector input;

        public LayorInfo(INeuralNetwork layer)
        {
            this.network = layer;
            this.rawoutput = new NVector(network.getNumberOfNeurons());
            this.output = new NVector(network.getNumberOfNeurons());
        }
    }

    @Override
    public NVector output(NVector input)
    {
        return output(0, input);
    }

    @Override
    public NVector rawoutput(NVector input)
    {
        return null;  //To change body of implemented methods use File | Settings | File Templates.
    }

    @Override
    public int getNumberOfNeurons()
    {
        return 3;
    }

    protected NVector output(int layer, NVector input)
    {
        LayorInfo layorInfo = aLayers[layer];
        INeuralNetwork network = aLayers[layer].network;

        //save info for back propagation

        layorInfo.input = new NVector(input);
        calculateRawOutput(layorInfo, network); //calculate v_k's
        calculateOutput(layorInfo, network);

        if (layer < aLayers.length - 1)
        {
            return output(layer+1, aLayers[layer].output);
        }
        return layorInfo.output;
    }

    private void calculateRawOutput(LayorInfo layorInfo, INeuralNetwork network)
    {
        //calculate v_k's
        int len=0;
        for(Neuron neuron:(SingleLayorNeuralNetwork)network)
        {
            layorInfo.rawoutput.set(len++, neuron.rawoutput(layorInfo.input));
        }
    }

    private void calculateOutput(LayorInfo layorInfo, INeuralNetwork network)
    {
        //calculate y_k's
        int len=0;
        for(Neuron neuron:(SingleLayorNeuralNetwork)network)
        {
            layorInfo.output.set(len, neuron.phi().apply(layorInfo.rawoutput.get(len++)));
        }
    }

/*    public void backpropagation()
    {
        adjustWeights(aLayers.length - 1);
    }

    protected void adjustWeights(int layer)
    {
        INeuralNetwork nLayer = aLayers[layer];
        for(Neuron neuron:(SingleLayorNeuralNetwork)nLayer)
        {
            //iterate thru the neuron's weights
            for(int weight=0; weight<neuron.size(); weight++)
            {
                float newWeight = neuron.getWeight(weight)
                                  + alpha * getPrevWeight(layer, weight)
                                  + eta * gradient(layer) * getPrevInput(layer, weight);
            }
        }

    }

    private float gradient(int layer, int weight)
    {
        if (layer == aLayers.length-1)
        {
            aError[weight] * phi.derivative()
        }
    }*/

}
