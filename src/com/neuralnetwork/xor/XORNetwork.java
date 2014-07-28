package com.neuralnetwork.xor;

import com.neuralnetwork.core.ActivationFunctions;
import com.neuralnetwork.core.interfaces.OldINeuralNetwork;
import com.neuralnetwork.core.neuron.NVector;
import com.neuralnetwork.core.neuron.Neuron;
import com.neuralnetwork.core.deprecated.SingleLayerNeuralNetwork;
import com.neuralnetwork.core.interfaces.IActivationFunction;

import java.util.Iterator;

/**
 * This neural network solves the XOR problem
 */
public class XORNetwork implements OldINeuralNetwork<NVector,NVector,Neuron>
{
    OldINeuralNetwork<NVector,NVector,Neuron>[] aLayers;

    public XORNetwork()
    {
        IActivationFunction phi = new ActivationFunctions.ThresholdFunction();
        OldINeuralNetwork firstPass = new SingleLayerNeuralNetwork();
        OldINeuralNetwork secondPass = new SingleLayerNeuralNetwork();

        ((SingleLayerNeuralNetwork)firstPass)
                .setNeurons(new Neuron(phi, 1f,1f,-1.5f),
                        new Neuron(phi, 1f,1f,-0.5f));

        ((SingleLayerNeuralNetwork)secondPass)
                .setNeurons(new Neuron(phi, -2f,1f,-0.5f));

        aLayers = new OldINeuralNetwork[2];
        aLayers[0] = firstPass;
        aLayers[1] = secondPass;
    }

    @Override
    public NVector generateOutput(NVector input)
    {
        return output(0, input);
    }

    @Override
    public NVector generateInducedLocalField(NVector input)
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
        NVector output = aLayers[layer].generateOutput(new NVector(input, 1f));
        if (layer < aLayers.length - 1)
        {
            return output(layer+1, output);
        }
        return output;
    }

    @Override
    public Neuron getNeuron(int neuron)
    {
        return null;  //To change body of implemented methods use File | Settings | File Templates.
    }

    @Override
    public Iterator<Neuron> iterator()
    {
        return null;  //To change body of implemented methods use File | Settings | File Templates.
    }
}
