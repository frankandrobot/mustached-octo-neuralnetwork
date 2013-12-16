package com.neuralnetwork.core.interfaces;

import com.neuralnetwork.core.NVector;
import com.neuralnetwork.core.Neuron;

public interface INeuralNetwork<T> extends Iterable<Neuron>
{
    public T output(NVector input);

    public T inducedLocalField(NVector input);

    public int getNumberOfNeurons();

    public Neuron getNeuron(int neuron);
}
