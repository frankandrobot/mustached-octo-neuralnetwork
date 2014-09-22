package com.neuralnetwork.cnn.map.builder;

import com.neuralnetwork.cnn.filter.IFilter;
import com.neuralnetwork.core.neuron.Neuron;

abstract class AbstractFeatureMapBuilder<Layer extends AbstractFeatureMapBuilder>
{
    protected int inputDim;

    protected Neuron sharedNeuron;

    protected IFilter filter;

    /**
     * Assumes a square input
     * @param inputSize
     * @return
     */
    public Layer set1DInputSize(int inputSize)
    {
        this.inputDim = inputSize;

        return (Layer) this;
    }

    public Layer setNeuron(Neuron neuron)
    {
        this.sharedNeuron = neuron;

        return (Layer) this;
    }

    public int getInputDim()
    {
        return inputDim;
    }

    public Neuron getSharedNeuron()
    {
        return sharedNeuron;
    }
}
