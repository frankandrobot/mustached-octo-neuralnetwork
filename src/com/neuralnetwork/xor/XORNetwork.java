package com.neuralnetwork.xor;

/**
 * This neural network solves the XOR problem
 */
public class XORNetwork implements INeuralNetwork
{
    INeuralNetwork[] aLayers;

    public XORNetwork()
    {
        IActivationFunction phi = new IActivationFunction.ThresholdFunction();
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
}
