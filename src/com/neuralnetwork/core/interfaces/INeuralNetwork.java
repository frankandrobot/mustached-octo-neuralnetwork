package com.neuralnetwork.core.interfaces;

import com.neuralnetwork.core.NVector;

public interface INeuralNetwork
{
    public NVector output(NVector input);

    public NVector inducedLocalField(NVector input);

    public int getNumberOfNeurons();
}
