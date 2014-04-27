package com.neuralnetwork.cnn.layers;

import com.neuralnetwork.cnn.MNeuron;
import com.neuralnetwork.cnn.filter.IConvolutionFilter;

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

    public int getInputDim() {
        return inputDim;
    }

    public MNeuron getSharedNeuron() {
        return sharedNeuron;
    }

    public IConvolutionFilter getConvolutionFilter() {
        return convolutionFilter;
    }
}
