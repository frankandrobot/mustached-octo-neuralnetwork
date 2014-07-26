package com.neuralnetwork.cnn.layer.builder;

import com.neuralnetwork.cnn.MNeuron;
import com.neuralnetwork.cnn.filter.IFilter;

public abstract class FeatureMapBuilder<T extends FeatureMapBuilder>
{
    protected int inputDim;

    protected MNeuron sharedNeuron;

    protected IFilter filter;

    /**
     * Assumes a square input
     * @param inputSize
     * @return
     */
    public T set1DInputSize(int inputSize)
    {
        this.inputDim = inputSize;
        return (T) this;
    }

    public T setNeuron(MNeuron neuron)
    {
        this.sharedNeuron = neuron;
        return (T) this;
    }

    public int getInputDim() {
        return inputDim;
    }

    public MNeuron getSharedNeuron() {
        return sharedNeuron;
    }
}
