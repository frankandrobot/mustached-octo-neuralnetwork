package com.neuralnetwork.xor;

import com.neuralnetwork.core.ActivationFunctions;
import com.neuralnetwork.core.NVector;
import com.neuralnetwork.core.Neuron;
import com.neuralnetwork.core.SingleLayerNeuralNetwork;
import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.INeuralNetwork;

/**
 * This neural network solves the XOR problem
 */
public class XORNetwork implements INeuralNetwork
{
    INeuralNetwork[] aLayers;

    public XORNetwork()
    {
        IActivationFunction phi = new ActivationFunctions.ThresholdFunction();
        INeuralNetwork firstPass = new SingleLayerNeuralNetwork();
        INeuralNetwork secondPass = new SingleLayerNeuralNetwork();

        ((SingleLayerNeuralNetwork)firstPass)
                .setNeurons(new Neuron(phi, 1f,1f,-1.5f),
                        new Neuron(phi, 1f,1f,-0.5f));

        ((SingleLayerNeuralNetwork)secondPass)
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

    @Override
    public NVector inducedLocalField(NVector input)
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
        //add bias to the end
        NVector output = aLayers[layer].output(new NVector(input, 1f));
        if (layer < aLayers.length - 1)
        {
            return output(layer+1, output);
        }
        return output;
    }
}
