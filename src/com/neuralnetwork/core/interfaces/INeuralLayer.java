package com.neuralnetwork.core.interfaces;

public interface INeuralLayer<T,N extends INeuron>
{
    public T generateOutput(T input);

    public T generateInducedLocalField(T input);

    public int getInputDim();

    public int getOutputDim();

    public N[] getNeurons();
}
