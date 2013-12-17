package com.neuralnetwork.core.interfaces;

public interface INeuralNetwork<I,O,N extends INeuron<?>> extends Iterable<N>
{
    public O output(I input);

    public O inducedLocalField(I input);

    public int getNumberOfNeurons();

    public N getNeuron(int neuron);
}
