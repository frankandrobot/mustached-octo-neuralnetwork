package com.neuralnetwork.xor;

public class XORNetwork implements INeuralNetwork
{
    INeuralNetwork firstPass;
    INeuralNetwork secondPass;

    public XORNetwork()
    {
        IActivationFunction phi = new IActivationFunction.ThresholdFunction();
        firstPass = new SingleLayorNeuralNetwork(2);
        secondPass = new SingleLayorNeuralNetwork(1);

        ((SingleLayorNeuralNetwork)firstPass)
                .setNeurons(new Neuron(phi, 1f,1f,-1.5f),
                            new Neuron(phi, 1f,1f,-0.5f));

        ((SingleLayorNeuralNetwork)secondPass)
                .setNeurons(new Neuron(phi, -2f,1f,-0.5f));
    }

    @Override
    public NVector output(NVector input)
    {
        NVector hiddenOutput = firstPass.output(input);
        return secondPass.output(hiddenOutput);
    }
}
