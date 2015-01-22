package com.neuralnetwork.cnn.singlelayer;

import com.neuralnetwork.cnn.filter.ISamplingFilter;
import com.neuralnetwork.core.neuron.Neuron;

public class SamplingSingleLayerBuilder extends AbstractCNNSingleLayerBuilder<SamplingSingleLayerBuilder,SamplingSingleLayer>
{
    public SamplingSingleLayerBuilder setFilter(ISamplingFilter... filter) {

        this.aFilters = filter;

        return this;
    }

    public ISamplingFilter[] getFilters()
    {
        return (ISamplingFilter[]) this.aFilters;
    }


    protected SamplingSingleLayer buildMap()
    {
        validate();

        return new SamplingSingleLayer(this);
    }

    private void validate()
    {
        if (inputDim % kernelDim != 0)
            throw new IllegalArgumentException("Input size must be a multiple of the kernel size");

        for(Neuron neuron:aSharedNeurons) {
            double[] weights = neuron.getWeightsWithoutBias();
            double weight = weights[0];

            for (int i = 1; i < weights.length; i++)
                if (weights[i] != weight)
                    throw new IllegalArgumentException("Kernel weights must all be equal");
        }
    }
}
