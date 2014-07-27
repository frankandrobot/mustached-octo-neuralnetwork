package com.neuralnetwork.core.interfaces;

public interface INeuralLayer<Input, Output>
{
    public Output generateOutput(Input input);

    public Output generateInducedLocalField(Input input);
}
