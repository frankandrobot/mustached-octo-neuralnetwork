package com.neuralnetwork.cnn.layer.builder;

import com.neuralnetwork.cnn.filter.IFilter;
import com.neuralnetwork.core.neuron.MNeuron;

abstract class FeatureMapBuilder<Layer extends FeatureMapBuilder>
{
    protected int inputDim;

    protected MNeuron sharedNeuron;

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

    public Layer setNeuron(MNeuron neuron)
    {
        this.sharedNeuron = neuron;

        return (Layer) this;
    }

    public int getInputDim()
    {
        return inputDim;
    }

    public MNeuron getSharedNeuron()
    {
        return sharedNeuron;
    }
}
