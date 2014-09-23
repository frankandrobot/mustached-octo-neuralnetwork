package com.neuralnetwork.core.interfaces;

public interface INeuralNetwork<Input,Output>
{
    public Output generateOutput(Input input);

    public Output generateYoutput(Input input);
}
