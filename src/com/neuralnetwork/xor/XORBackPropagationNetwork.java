package com.neuralnetwork.xor;

public class XORBackPropagationNetwork implements INeuralNetwork
{
    INeuralNetwork[] aLayers;

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

        aLayers = new INeuralNetwork[2];
        aLayers[0] = firstPass;
        aLayers[1] = secondPass;
    }

    @Override
    public NVector output(NVector input)
    {
        return output(0, input);
    }

    protected NVector output(int layer, NVector input)
    {
        NVector output = aLayers[layer].output(input);
        if (layer < aLayers.length - 1)
        {
            return output(layer+1, output);
        }
        return output;
    }

    public void backpropagation()
    {
        adjustWeights(aLayers.length - 1);
    }

    protected void adjustWeights(int layer)
    {
        INeuralNetwork nLayer = aLayers[layer];
        for(Neuron neuron:(SingleLayorNeuralNetwork)nLayer)
        {
            for(int weight=0; weight<neuron.size(); weight++)
            {
                float newWeight = neuron.getCurWeight(weight)
                                  + alpha * neuron.getPrevWeight(weight)
                                  + eta * gradient(layer) * neuron.getPrevInput(weight);
            }
        }

    }

    private float gradient(int layer, int weight)
    {
        if (layer == aLayers.length-1)
        {
            aError[weight] * phi.derivative()
        }
    }

}
