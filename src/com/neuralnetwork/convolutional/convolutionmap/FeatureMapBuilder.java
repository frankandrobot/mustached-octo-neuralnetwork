package com.neuralnetwork.convolutional.convolutionmap;

import com.neuralnetwork.convolutional.MNeuron;
import com.neuralnetwork.convolutional.filter.IConvolutionFilter;

public class FeatureMapBuilder
{
    protected int inputDim;
    protected MNeuron sharedNeuron;
    protected IConvolutionFilter convolutionFilter;

    /**
     * Assumes a square input
     * @param inputSize
     * @return
     */
    public FeatureMapBuilder set1DInputSize(int inputSize)
    {
        this.inputDim = inputSize;
        return this;
    }

    public FeatureMapBuilder setNeuron(MNeuron neuron)
    {
        this.sharedNeuron = neuron;
        return this;
    }

    public FeatureMapBuilder setConvolutionFilter(IConvolutionFilter filter)
    {
        this.convolutionFilter = filter;
        return this;
    }
}
