package com.neuralnetwork.xor;

public interface INeuralNetwork
{
    public NVector output(NVector input);

    public NVector inducedLocalField(NVector input);

    public int getNumberOfNeurons();
}
