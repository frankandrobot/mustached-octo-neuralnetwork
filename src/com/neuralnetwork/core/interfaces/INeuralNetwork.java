package com.neuralnetwork.core.interfaces;

import com.neuralnetwork.core.NVector;
import com.neuralnetwork.core.Neuron;

public interface INeuralNetwork extends Iterable<Neuron>
{
    public NVector output(NVector input);

    public NVector inducedLocalField(NVector input);

    public int getNumberOfNeurons();

    public Neuron getNeuron(int neuron);
}
