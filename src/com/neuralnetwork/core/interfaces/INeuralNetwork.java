package com.neuralnetwork.core.interfaces;

import com.neuralnetwork.core.Neuron;

public interface INeuralNetwork<I,O> extends Iterable<Neuron>
{
    public O output(I input);

    public O inducedLocalField(I input);

    public int getNumberOfNeurons();

    public Neuron getNeuron(int neuron);
}
